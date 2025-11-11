# 🧭 General Coding & Review Guidelines

## 1. Code Review Objectives

When reviewing or developing a new module:

- **Cleanliness:** Identify and remove unused variables, imports, and functions.
- **Modularity:** Move reusable or generic logic into helper modules.
- **Consistency:** Align structure, formatting, and signatures with other modules.
- **Compactness:** Simplify code without reducing clarity or robustness.
- **Integration:** Ensure the module fits naturally within both `openamundsen_da` and `openamundsen`.

### Review Questions

- Can any part be refactored or simplified?
- Are there duplicated or redundant sections?
- Should any logic be centralized (e.g., in `helpers`, `stats`, or `constants`)?
- Does the module follow our established structure and formatting conventions?
- Is the configuration handled consistently and externally defined where possible?

List fo helper functions:

#### Core and IO

- C:\Daten\PhD\openamundsen_da\openamundsen_da\core\constants.py
- C:\Daten\PhD\openamundsen_da\openamundsen_da\core\config.py
- C:\Daten\PhD\openamundsen_da\openamundsen_da\core\env.py
- C:\Daten\PhD\openamundsen_da\openamundsen_da\io\paths.py

#### Utilities

- C:\Daten\PhD\openamundsen_da\openamundsen_da\util\ts.py
- C:\Daten\PhD\openamundsen_da\openamundsen_da\util\stats.py
- C:\Daten\PhD\openamundsen_da\openamundsen_da\util\aoi.py

#### Viz helpers

- C:\Daten\PhD\openamundsen_da\openamundsen_da\methods\viz_style.py
- C:\Daten\PhD\openamundsen_da\openamundsen_da\methods\viz_utils.py

---

## 2. Code Design Principles

- Write **compact, readable, and modular** code.
- Ensure **all variables, constants, and functions are used**.
- Avoid **duplicate code** – consolidate shared logic in helper scripts.
- Keep the code **robust, extendable, and maintainable** for future additions.
- Use **type hints** and **explicit function signatures**.
- Follow **openAMUNDSEN-style conventions** for naming, structure, and error handling.
- Prioritize **clarity over cleverness** – the code should be self-explanatory.

---

## 3. Logging

- Use **`loguru`** for all logging.
- Apply the standard format defined in `constants.py`:
  ```python
  format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level:<8}</level> | {message}"
  ```
