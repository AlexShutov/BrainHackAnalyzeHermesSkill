# Corpus enrichment: sourced definitions and examples per device

Goal: supplement the closed device list with facts that are NOT already in its definition, plus a real usage example per device, each traceable to a source.

## Procedure

1. **Two queries per item**, in this order:
   - native language: `"<device name>" приведи примеры` (the explicit "give examples" clause is what pulls example lists rather than another definition);
   - English: `"<established English term>" examples` — search by the accepted term, not by the skill's own label. Own coinages ("cumulative framing", "deferred conclusion", "pre-emptive inoculation") return nothing under their own names; the standard equivalents (framing cascade, enthymeme, inoculation/prebunking) are what carry the literature.
2. **Dedupe against the existing definition** by token overlap, computed in code, not by eye: below ~0.45 it is a new fact, above ~0.6 it is the same sentence in other words. Store the two kinds separately (`extra` vs `examples`) so a restatement never masquerades as new knowledge.
3. **Classify each hit**: a real usage example is quoted speech (a phrase the manipulator actually wrote). A definition that merely contains quotes is not an example — it stays as theory. Only real usage examples enter the eval set.
4. **Keep the enrichment in its own file** (`psychology/devices_enriched.json`: `en_name`, `examples` with source URL, `extra`). Never overwrite the curated definitions in `devices.json` — the closed list is the authority, the enrichment is the supplement.
5. **Batch through code**: fan out searches in an `execute_code` loop, append raw results to a workspace JSON, then aggregate/dedupe/count programmatically. Verify the coverage count against the target before reporting it.

## Pitfalls

- **Misattribution from bulk extraction.** A snippet fetched for one device lands under several when the source page lists many devices in one block — the same quote can end up attached to five different entries. Before writing an example into a dataset, check that the quote illustrates THAT device; if it came from a list page, re-read the surrounding lines. Wrong attribution in an eval set silently corrupts every future measurement.
- **Fragments, not examples.** Headings, code, navigation text and truncated sentences sneak into extracted snippets. Drop anything that is not a complete sentence the author wrote.
- **Forcing coverage.** A device with only a definition and no usage example must stay uncovered and be reported as such — an invented example is worse than a gap. Report the uncovered list explicitly.
- **Citing a snippet as if the page was read.** A search description supports only what it literally says; for a load-bearing example, extract the page and quote from it (see `grounded-citations`).
- **Engine availability changes the route, not the method.** If one search engine refuses the browser, the configured search backend and alternate engines still serve the same procedure — do not stop the enrichment and do not report the tool as broken.
