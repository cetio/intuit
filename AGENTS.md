# Verification

- Build the library: `dub build --compiler=ldc2 --config=library`.
- Run unit tests: `dub run --compiler=ldc2 --config=unittest --build=unittest`.
- The `integration` configuration enables tests requiring provider credentials or local model servers; use `unittest` for offline verification.
