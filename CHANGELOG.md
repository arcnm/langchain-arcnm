# Changelog

## 0.1.1

- `precheck_part`: before pricing a STEP file, see whether it is one part or an
  assembly, its bill of materials, and how many calculations pricing it uses.
  Free; starts nothing. Needs `parts:write`, so `read_only=True` leaves it out.
- `get_cost_sheet` returns each line's amount, time, time basis and rate, and
  the lot-size point, without the message ids, formulas and evidence the ARCNM
  page uses. The figures are unchanged.

## 0.1.0

First release. Price a part from a CAD file or a part revision, read and list
calculations, read the cost sheet at any lot size, find parts and their
revisions, list costing environments, check the calculation allowance, and
cancel a running calculation.
