# Changelog

All notable user-visible changes to USDCraft are recorded here.

## 1.0.0 - 2026-10-30

First release.

- Generation: create an articulated, simulator-ready asset from text and/or one reference image.
- Reconstruction: rebuild one real object from a source mesh plus one photo of it.
- Batch runs from a CSV for both products.
- Two ways to run the agent: the Provider (the OpenAI API by default) and Codex.
- Export to URDF, USD and USDZ with PhysX and Newton physics.
- Local Viewer to browse, inspect, rate and download results.
- Example inputs for both products in `assets/`.
- `THIRD_PARTY_NOTICES.md`: the inventory of all runtime, bundled and build-time dependencies and
  install-time downloads, with their license texts.
