import { describe, expect, it } from "vitest";
import {
  GGML_MAGIC,
  HEADER_BYTES,
  checkHeader,
  describeFtype,
  inspect,
  parseHeader,
} from "./index.js";

/**
 * Builds a header from twelve integers. The defaults match a measured, working
 * file: whisper-base, F16, vocabulary 51865.
 */
function header(
  overrides: Partial<Record<string, number>> = {},
): Uint8Array {
  const fields = {
    magic: GGML_MAGIC,
    nVocab: 51865,
    nAudioCtx: 1500,
    nAudioState: 512,
    nAudioHead: 8,
    nAudioLayer: 6,
    nTextCtx: 448,
    nTextState: 512,
    nTextHead: 8,
    nTextLayer: 6,
    nMels: 80,
    ftype: 1,
    ...overrides,
  };

  const bytes = new Uint8Array(HEADER_BYTES);
  const view = new DataView(bytes.buffer);
  Object.values(fields).forEach((value, i) => view.setUint32(i * 4, value >>> 0, true));
  return bytes;
}

describe("parseHeader", () => {
  it("reads all twelve fields", () => {
    const h = parseHeader(header());
    expect(h.magic).toBe(GGML_MAGIC);
    expect(h.nVocab).toBe(51865);
    expect(h.nTextCtx).toBe(448);
    expect(h.nMels).toBe(80);
    expect(h.ftype).toBe(1);
  });

  it("reads n_text_ctx at byte offset 24", () => {
    const bytes = header({ nTextCtx: 448 });
    const own = new DataView(bytes.buffer).getInt32(24, true);
    expect(own).toBe(448);
  });

  it("insists on enough bytes", () => {
    expect(() => parseHeader(new Uint8Array(20))).toThrow(RangeError);
  });

  it("copes with a slice of a larger buffer", () => {
    // This is how it is read in practice: the first 48 bytes of a large file.
    const large = new Uint8Array(4096);
    large.set(header(), 0);
    expect(parseHeader(large.subarray(0, HEADER_BYTES)).nTextCtx).toBe(448);
  });
});

describe("describeFtype", () => {
  it("recognises unquantised types", () => {
    expect(describeFtype(1)).toMatchObject({ quantisationVersion: 0, label: "f16" });
    expect(describeFtype(0).label).toBe("f32");
  });

  it("splits the quantised value into version and base type", () => {
    // Measured against real files: a base conversion quantised to q5_0 reports
    // 2008, the same one quantised to q8_0 reports 2007. So in the ggml enum
    // 7 is q8_0 and 8 is q5_0, not the other way round.
    expect(describeFtype(2008)).toMatchObject({
      quantisationVersion: 2,
      baseType: 8,
      label: "q5_0 (qnt v2)",
    });
    expect(describeFtype(2007).label).toBe("q8_0 (qnt v2)");
  });

  it("names the unknown as unknown", () => {
    expect(describeFtype(999).label).toContain("unknown");
  });
});

describe("checkHeader", () => {
  it("reports nothing for a healthy file", () => {
    expect(checkHeader(parseHeader(header()))).toEqual([]);
  });

  it("catches the common 1024 mis-conversion and names the cause", () => {
    const f = checkHeader(parseHeader(header({ nTextCtx: 1024 })));
    expect(f).toHaveLength(1);
    expect(f[0].severity).toBe("blocking");
    expect(f[0].message).toContain("max_target_positions");
  });

  it("reports any other wrong value as blocking too", () => {
    const f = checkHeader(parseHeader(header({ nTextCtx: 512 })));
    expect(f[0].severity).toBe("blocking");
    expect(f[0].message).toContain("512");
  });

  it("recognises a file that is not GGML at all and then stops", () => {
    const f = checkHeader(parseHeader(header({ magic: 0x46554747 })));
    expect(f).toHaveLength(1);
    expect(f[0].field).toBe("magic");
    expect(f[0].message).toContain("GGUF");
  });

  it("accepts 128 mel bands for large-v3", () => {
    expect(checkHeader(parseHeader(header({ nMels: 128, nVocab: 51866 })))).toEqual([]);
  });

  it("rejects an unknown mel count as blocking", () => {
    const f = checkHeader(parseHeader(header({ nMels: 64 })));
    expect(f.some((x: { field: string; severity: string }) => x.field === "nMels" && x.severity === "blocking")).toBe(true);
  });

  it("warns about a foreign vocabulary without blocking", () => {
    const f = checkHeader(parseHeader(header({ nVocab: 32000 })));
    expect(f).toHaveLength(1);
    expect(f[0].severity).toBe("warning");
  });
});

describe("inspect", () => {
  it("calls a healthy file loadable", () => {
    const r = inspect(header({ ftype: 2008 }));
    expect(r.loadable).toBe(true);
    expect(r.findings).toEqual([]);
    expect(r.header.nTextCtx).toBe(448);
  });

  it("calls the 1024 file not loadable", () => {
    expect(inspect(header({ nTextCtx: 1024 })).loadable).toBe(false);
  });

  it("stays loadable when there is only a warning", () => {
    expect(inspect(header({ nAudioCtx: 1000 })).loadable).toBe(true);
  });
});
