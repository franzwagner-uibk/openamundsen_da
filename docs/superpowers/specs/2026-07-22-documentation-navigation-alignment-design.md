---
published: false
---

# Documentation navigation alignment design

## Goal

Align the visible openAMUNDSEN-DA documentation structure with the established
openAMUNDSEN documentation structure while preserving every existing public URL
and all technical content.

## Scope

The change is limited to rendered navigation, page titles, headings and visible
internal-link labels. It does not rename or move files, change permalinks,
modify commands or examples, or restructure tutorial, advanced or reference
content.

## Navigation

The rendered top-level navigation will be:

1. Home
2. Documentation
   1. Installation
   2. Input data
   3. Configuration
   4. Running the model
   5. Output data
   6. Example data sets
3. How to Use
4. Advanced
5. Reference
6. openAMUNDSEN (external link)

`Documentation` will be a short landing page linking to its six child pages.
The data-assimilation-specific `How to Use`, `Advanced` and `Reference` sections
remain separate top-level entries.

## Page mapping and compatibility

| Existing source | Visible title after alignment | Existing URL retained |
| --- | --- | --- |
| `installation.md` | Installation | `/installation.html` |
| `guides/observations.md` | Input data | `/guides/observations.html` |
| `guides/configuration.md` | Configuration | `/guides/configuration.html` |
| `running.md` | Running the model | `/running.html` |
| `output-data.md` | Output data | `/output-data.html` |
| `example-data.md` | Example data sets | `/example-data.html` |

The six pages receive `parent: Documentation` and child-local navigation order
values matching openAMUNDSEN. Visible references to these page names are aligned
throughout the documentation, without changing their link targets.

## Validation

- Run the documentation contract validator and production Jekyll build.
- Confirm the generated navigation hierarchy and retained URLs.
- Check all internal links and assets.
- Open the landing page and all six child pages in a real browser and confirm
  that there are no page or console errors.
