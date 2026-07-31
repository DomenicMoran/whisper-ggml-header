/**
 * Reads and checks the header of a Whisper GGML model file.
 *
 * Why this exists: whisper.cpp and its bindings refuse to load a model whose
 * `n_text_ctx` is not 448, and the error surfaces as a generic "model
 * unavailable". The value is wrong in a large share of community conversions,
 * because the upstream conversion script writes `max_length` from the Hugging
 * Face `config.json` into that slot, and several fine-tunes carry
 * `max_length: 1024` there as a *generation* parameter. The real decoder
 * context length is `max_target_positions`, which is 448.
 *
 * The check costs 48 bytes and a few microseconds. Finding out at runtime, on
 * a user's device, costs considerably more.
 */

/** The twelve int32 fields at the start of a Whisper GGML file. */
export type WhisperGgmlHeader = {
  /** `0x67676d6c`, the ASCII bytes "ggml" in little-endian order. */
  magic: number;
  nVocab: number;
  nAudioCtx: number;
  nAudioState: number;
  nAudioHead: number;
  nAudioLayer: number;
  /** Must be 448. This is the field that decides whether the model loads. */
  nTextCtx: number;
  nTextState: number;
  nTextHead: number;
  nTextLayer: number;
  nMels: number;
  /**
   * Raw file type. Quantised files encode it as
   * `quantisationVersion * 1000 + baseType`, so a q5_0 file at quantisation
   * version 2 reads as 2008.
   */
  ftype: number;
};

export const GGML_MAGIC = 0x67676d6c;

/** The context length whisper.cpp requires. Not configurable, by design. */
export const REQUIRED_TEXT_CTX = 448;

/** Number of bytes the header occupies: twelve 32-bit integers. */
export const HEADER_BYTES = 48;

/**
 * The `ggml_ftype` enum, in the order ggml actually defines it.
 *
 * Worth reading twice: 7 is q8_0 and 8 is q5_0, not the other way round, and
 * 5 and 6 do not exist. Both facts were verified against real files rather
 * than assumed. A base-model conversion quantised to q5_0 reports ftype 2008
 * and one quantised to q8_0 reports 2007, with the q5_0 file being the smaller
 * of the two (55 MB against 82 MB).
 */
const BASE_TYPES: Record<number, string> = {
  0: "f32",
  1: "f16",
  2: "q4_0",
  3: "q4_1",
  4: "q4_1_some_f16",
  7: "q8_0",
  8: "q5_0",
  9: "q5_1",
  10: "q2_k",
  11: "q3_k",
  12: "q4_k",
  13: "q5_k",
  14: "q6_k",
};

/**
 * Splits a raw `ftype` into its two parts.
 *
 * Values below 1000 are unquantised and mean the base type directly. Above
 * that, the thousands digit carries the quantisation format version.
 */
export function describeFtype(ftype: number): {
  quantisationVersion: number;
  baseType: number;
  label: string;
} {
  const quantisationVersion = Math.floor(ftype / 1000);
  const baseType = ftype % 1000;
  const name = BASE_TYPES[baseType] ?? `unknown(${baseType})`;
  return {
    quantisationVersion,
    baseType,
    label: quantisationVersion > 0 ? `${name} (qnt v${quantisationVersion})` : name,
  };
}

/**
 * Parses the header out of the first bytes of a model file.
 *
 * @param bytes at least the first 48 bytes of the file. Passing the whole file
 *   works too, but is never necessary.
 * @throws if fewer than 48 bytes are given.
 */
export function parseHeader(bytes: Uint8Array): WhisperGgmlHeader {
  if (bytes.length < HEADER_BYTES) {
    throw new RangeError(
      `Need at least ${HEADER_BYTES} bytes to read the header, got ${bytes.length}.`,
    );
  }

  const view = new DataView(bytes.buffer, bytes.byteOffset, HEADER_BYTES);
  const i32 = (index: number) => view.getInt32(index * 4, true);

  return {
    magic: view.getUint32(0, true),
    nVocab: i32(1),
    nAudioCtx: i32(2),
    nAudioState: i32(3),
    nAudioHead: i32(4),
    nAudioLayer: i32(5),
    nTextCtx: i32(6),
    nTextState: i32(7),
    nTextHead: i32(8),
    nTextLayer: i32(9),
    nMels: i32(10),
    ftype: i32(11),
  };
}

export type Finding = {
  field: string;
  message: string;
  /** `blocking` means whisper.cpp will not load this file. */
  severity: "blocking" | "warning";
};

/**
 * Checks a parsed header for the problems that actually occur in the wild.
 *
 * Returns an empty array for a sound file. Everything reported here is either
 * a refusal to load or a strong sign that the conversion went wrong.
 */
export function checkHeader(header: WhisperGgmlHeader): Finding[] {
  const findings: Finding[] = [];

  if (header.magic !== GGML_MAGIC) {
    findings.push({
      field: "magic",
      severity: "blocking",
      message: `Not a GGML file: magic is 0x${header.magic.toString(16)}, expected 0x${GGML_MAGIC.toString(16)}. A GGUF file starts with "GGUF" and needs a different loader.`,
    });
    // Everything after this would be noise.
    return findings;
  }

  if (header.nTextCtx !== REQUIRED_TEXT_CTX) {
    findings.push({
      field: "nTextCtx",
      severity: "blocking",
      message:
        header.nTextCtx === 1024
          ? "n_text_ctx is 1024, the classic broken conversion. The script took max_length from config.json instead of max_target_positions. whisper.cpp will not load this file."
          : `n_text_ctx is ${header.nTextCtx}, expected ${REQUIRED_TEXT_CTX}. whisper.cpp will not load this file.`,
    });
  }

  if (header.nMels !== 80 && header.nMels !== 128) {
    findings.push({
      field: "nMels",
      severity: "blocking",
      message: `n_mels is ${header.nMels}, expected 80 (or 128 for large-v3). The mel filter bank does not match the model.`,
    });
  }

  // 51865 is the multilingual vocabulary, 51866 is large-v3, 51864 is the
  // English-only build. Anything else means the tokenizer got mixed up.
  if (![51864, 51865, 51866].includes(header.nVocab)) {
    findings.push({
      field: "nVocab",
      severity: "warning",
      message: `n_vocab is ${header.nVocab}, which is none of the known Whisper vocabularies (51864, 51865, 51866).`,
    });
  }

  if (header.nAudioCtx !== 1500) {
    findings.push({
      field: "nAudioCtx",
      severity: "warning",
      message: `n_audio_ctx is ${header.nAudioCtx}, expected 1500 (30 seconds of audio).`,
    });
  }

  const { baseType } = describeFtype(header.ftype);
  if (!(baseType in BASE_TYPES)) {
    findings.push({
      field: "ftype",
      severity: "warning",
      message: `Unknown file type ${header.ftype}.`,
    });
  }

  return findings;
}

/** Convenience: parse and check in one call. */
export function inspect(bytes: Uint8Array): {
  header: WhisperGgmlHeader;
  findings: Finding[];
  loadable: boolean;
} {
  const header = parseHeader(bytes);
  const findings = checkHeader(header);
  return {
    header,
    findings,
    loadable: !findings.some((f) => f.severity === "blocking"),
  };
}
