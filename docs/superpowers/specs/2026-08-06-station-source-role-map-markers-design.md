---
published: false
---

# Station Source and Role Map Markers

Status: approved by the user on 2026-08-06.

## Design

Project-map panels may opt into `station_marker_mode: sources_and_roles` without
changing the existing forcing-only default. The renderer loads forcing stations
from setup meteo and snow stations from the project-configured
`stations_da_metadata.csv`, matches each snow record independently to its
nearest forcing station within a configurable positive tolerance and renders
forcing, snow, combined and holdout markers. Multiple snow records at one
forcing site remain visible in a deterministic five-point circular fan.

Combined markers use a vertical red-left/blue-right split. Holdouts use a black
`x` marker that overrides source colors and is repeated consistently in both
inside-panel and below-panel station keys. Holdout symbols use marker area 18
instead of the standard station area 26. On maps they render two z-order levels
above every forcing, snow-only and split marker, independent of record order or
co-location. A `station_categories` legend item provides the fixed four-entry
key. Overview panels may independently enable full subdomain ID labels. The
renderer places them near representative interior points and separates their
text boxes where space permits. Existing panel dimensions and legacy map
recipes remain unchanged.

The North Tyrol generator enriches finalized snow-station role metadata with
coordinates, enables classified markers on its ROI and hillshade panels,
enables full IDs on the country overview and removes its former ROI label.
Country geometry continues to use the existing GISCO cache contract.
The North Tyrol below-panel station key abbreviates only its snow entry to
`Snow obs. station` and uses a slightly larger gap between its two rows; the
inside-panel key retains the full wording.

## Acceptance

Focused tests cover configuration validation, deterministic many-to-one
matching, circular offsets, split and role colors, the legend, subdomain labels
and missing metadata. Lenovo P8 rendering uses reviewed source commits and a
temporary setup before merge; no model propagation or package release occurs.
