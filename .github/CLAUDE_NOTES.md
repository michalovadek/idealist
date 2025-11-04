# Notes for Claude Code

## Documentation Guidelines

### README.md Structure
**CRITICAL: Keep the README concise!**

The README should ONLY contain:
1. Brief description (2-3 sentences)
2. Basic installation (`pip install idealist`)
3. Quick start example (5-10 lines of code)
4. Link to documentation
5. Link to examples directory

**DO NOT add to README:**
- ❌ Detailed installation options (GPU, TPU, etc.)
- ❌ Troubleshooting sections
- ❌ Performance tuning tips
- ❌ Advanced configuration options
- ❌ Multiple examples
- ❌ Comprehensive feature lists

### Where to Document Features

Instead, use:
- **`docs/installation.md`** - Detailed installation for CPU/GPU/TPU
- **`docs/troubleshooting.md`** - Common issues and solutions
- **`docs/configuration.md`** - Device config, CPU cores, etc.
- **`docs/api.md`** - API reference
- **`examples/`** - Working code examples

### Golden Rule
**If you're tempted to add a section to README, create a separate doc file instead and link to it.**

The README should be scannable in under 30 seconds.

## Git Repository Hygiene

### Keep the Root Clean
**CRITICAL: Do not clutter the main git folder with temporary files!**

**NEVER commit these to the repository:**
- ❌ Analysis files (e.g., `RESPONSE_TYPE_ANALYSIS.txt`)
- ❌ Workflow check scripts (e.g., `check_workflows.py`)
- ❌ Temporary notes or scratch files
- ❌ Debug outputs or logs
- ❌ Personal testing scripts

### Proper Handling of Temporary Files

1. **Create temporary files in ignored locations:**
   - Use `/experiments/` for experimental code
   - Use `/results/` for analysis outputs
   - These are already in `.gitignore`

2. **If you must create temporary files in root:**
   - Immediately add them to `.gitignore`
   - Delete them when done
   - Use patterns like `*.analysis.txt` or `*.notes.txt`

3. **Files that SHOULD be tracked:**
   - ✅ Source code (`idealist/`)
   - ✅ Tests (`tests/`)
   - ✅ Documentation (`docs/`)
   - ✅ Configuration (`pyproject.toml`, `.github/workflows/`)
   - ✅ Examples (`examples/`)

### Before Any Commit
Always check `git status` and ensure only intentional files are being committed.
