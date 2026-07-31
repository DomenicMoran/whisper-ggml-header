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
 * Baut einen Header aus zwölf Ganzzahlen. Die Vorgabewerte entsprechen einer
 * gemessenen, funktionierenden Datei: whisper-base, F16, Vokabular 51865.
 */
function header(
  ueberschreiben: Partial<Record<string, number>> = {},
): Uint8Array {
  const felder = {
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
    ...ueberschreiben,
  };

  const bytes = new Uint8Array(HEADER_BYTES);
  const view = new DataView(bytes.buffer);
  Object.values(felder).forEach((wert, i) => view.setUint32(i * 4, wert >>> 0, true));
  return bytes;
}

describe("parseHeader", () => {
  it("liest alle zwölf Felder", () => {
    const h = parseHeader(header());
    expect(h.magic).toBe(GGML_MAGIC);
    expect(h.nVocab).toBe(51865);
    expect(h.nTextCtx).toBe(448);
    expect(h.nMels).toBe(80);
    expect(h.ftype).toBe(1);
  });

  it("liest n_text_ctx bei Byte-Offset 24", () => {
    const bytes = header({ nTextCtx: 448 });
    const eigen = new DataView(bytes.buffer).getInt32(24, true);
    expect(eigen).toBe(448);
  });

  it("verlangt genug Bytes", () => {
    expect(() => parseHeader(new Uint8Array(20))).toThrow(RangeError);
  });

  it("kommt mit einem Ausschnitt eines größeren Puffers zurecht", () => {
    // So liest man in der Praxis: die ersten 48 Bytes einer großen Datei.
    const gross = new Uint8Array(4096);
    gross.set(header(), 0);
    expect(parseHeader(gross.subarray(0, HEADER_BYTES)).nTextCtx).toBe(448);
  });
});

describe("describeFtype", () => {
  it("erkennt unquantisierte Typen", () => {
    expect(describeFtype(1)).toMatchObject({ quantisationVersion: 0, label: "f16" });
    expect(describeFtype(0).label).toBe("f32");
  });

  it("zerlegt den quantisierten Wert in Version und Grundtyp", () => {
    // An echten Dateien nachgemessen: eine mit q5_0 quantisierte
    // base-Konvertierung meldet 2008, dieselbe mit q8_0 meldet 2007.
    // In der ggml-Aufzaehlung ist 7 also q8_0 und 8 ist q5_0, nicht umgekehrt.
    expect(describeFtype(2008)).toMatchObject({
      quantisationVersion: 2,
      baseType: 8,
      label: "q5_0 (qnt v2)",
    });
    expect(describeFtype(2007).label).toBe("q8_0 (qnt v2)");
  });

  it("benennt Unbekanntes als solches", () => {
    expect(describeFtype(999).label).toContain("unknown");
  });
});

describe("checkHeader", () => {
  it("meldet nichts bei einer gesunden Datei", () => {
    expect(checkHeader(parseHeader(header()))).toEqual([]);
  });

  it("erkennt die verbreitete 1024er-Fehlkonvertierung und benennt die Ursache", () => {
    const f = checkHeader(parseHeader(header({ nTextCtx: 1024 })));
    expect(f).toHaveLength(1);
    expect(f[0].severity).toBe("blocking");
    expect(f[0].message).toContain("max_target_positions");
  });

  it("meldet jeden anderen falschen Wert ebenfalls blockierend", () => {
    const f = checkHeader(parseHeader(header({ nTextCtx: 512 })));
    expect(f[0].severity).toBe("blocking");
    expect(f[0].message).toContain("512");
  });

  it("erkennt eine Datei, die gar kein GGML ist, und hört dann auf", () => {
    const f = checkHeader(parseHeader(header({ magic: 0x46554747 })));
    expect(f).toHaveLength(1);
    expect(f[0].field).toBe("magic");
    expect(f[0].message).toContain("GGUF");
  });

  it("akzeptiert 128 Mel-Bänder für large-v3", () => {
    expect(checkHeader(parseHeader(header({ nMels: 128, nVocab: 51866 })))).toEqual([]);
  });

  it("beanstandet eine unbekannte Mel-Zahl blockierend", () => {
    const f = checkHeader(parseHeader(header({ nMels: 64 })));
    expect(f.some((x: { field: string; severity: string }) => x.field === "nMels" && x.severity === "blocking")).toBe(true);
  });

  it("warnt bei fremdem Vokabular, ohne zu blockieren", () => {
    const f = checkHeader(parseHeader(header({ nVocab: 32000 })));
    expect(f).toHaveLength(1);
    expect(f[0].severity).toBe("warning");
  });
});

describe("inspect", () => {
  it("nennt eine gesunde Datei ladbar", () => {
    const r = inspect(header({ ftype: 2008 }));
    expect(r.loadable).toBe(true);
    expect(r.findings).toEqual([]);
    expect(r.header.nTextCtx).toBe(448);
  });

  it("nennt die 1024er-Datei nicht ladbar", () => {
    expect(inspect(header({ nTextCtx: 1024 })).loadable).toBe(false);
  });

  it("bleibt ladbar, wenn nur eine Warnung vorliegt", () => {
    expect(inspect(header({ nAudioCtx: 1000 })).loadable).toBe(true);
  });
});
