#!/usr/bin/env node
/**
 * Command line front end.
 *
 * Reads only the first 48 bytes of each file, so pointing it at a directory of
 * multi-gigabyte models costs nothing. Exit code 1 when any file would fail to
 * load, which makes it usable as a build step before publishing a model.
 */

import { open } from "node:fs/promises";
import { HEADER_BYTES, describeFtype, inspect } from "./index.js";

async function readHeaderBytes(path: string): Promise<Uint8Array> {
  const file = await open(path, "r");
  try {
    const buffer = new Uint8Array(HEADER_BYTES);
    const { bytesRead } = await file.read(buffer, 0, HEADER_BYTES, 0);
    if (bytesRead < HEADER_BYTES) {
      throw new RangeError(`${path}: file is only ${bytesRead} bytes.`);
    }
    return buffer;
  } finally {
    await file.close();
  }
}

async function main() {
  const paths = process.argv.slice(2);

  if (paths.length === 0 || paths[0] === "--help" || paths[0] === "-h") {
    console.log(
      [
        "whisper-ggml-header <file.bin> [more.bin ...]",
        "",
        "Prints the header of a Whisper GGML model and reports whether",
        "whisper.cpp would load it. Reads 48 bytes per file.",
        "",
        "Exit code 1 if any file is not loadable.",
      ].join("\n"),
    );
    process.exit(paths.length === 0 ? 1 : 0);
  }

  let schlecht = 0;

  for (const path of paths) {
    let ergebnis;
    try {
      ergebnis = inspect(await readHeaderBytes(path));
    } catch (error) {
      console.error(`${path}\n  cannot read: ${(error as Error).message}\n`);
      schlecht++;
      continue;
    }

    const { header, findings, loadable } = ergebnis;
    const ftype = describeFtype(header.ftype);

    console.log(path);
    console.log(`  n_text_ctx    ${header.nTextCtx}${header.nTextCtx === 448 ? "" : "   <- wrong"}`);
    console.log(`  n_vocab       ${header.nVocab}`);
    console.log(`  n_mels        ${header.nMels}`);
    console.log(`  n_audio_ctx   ${header.nAudioCtx}`);
    console.log(`  n_text_layer  ${header.nTextLayer}`);
    console.log(`  ftype         ${header.ftype} (${ftype.label})`);

    for (const f of findings) {
      console.log(`  ${f.severity === "blocking" ? "BLOCKING" : "warning "} ${f.field}: ${f.message}`);
    }

    console.log(`  => ${loadable ? "loadable" : "whisper.cpp will refuse this file"}\n`);
    if (!loadable) schlecht++;
  }

  process.exit(schlecht > 0 ? 1 : 0);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
