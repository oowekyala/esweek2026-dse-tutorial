"""A configuration picker that only offers feasible choices.

Most combinations of parameter values are infeasible -- of the 4 billion
combinations in the small space, 3234 survive the constraints -- so a free-form
form would mostly produce rejected guesses. The picker knows the feasible set
and, as each parameter is fixed, narrows every other dropdown to the values
that can still complete a feasible configuration.
"""

from __future__ import annotations

from typing import Callable

import ipywidgets as widgets
import pandas as pd
from IPython.display import display

from .cinm import iter_param_groups, ordering_labels, param_columns

ANY = "—"


class ConfigPicker:
    """Dropdowns per parameter, narrowed to what stays feasible.

    ``on_pick`` runs when the Simulate button is pressed with a complete
    configuration; whatever it returns is shown under the picker and kept as
    ``result``. Doing the evaluation from the button rather than from the next
    cell keeps it bound to the widget the user actually clicked in -- a re-run
    of the cell that created the picker would otherwise leave the displayed
    widget and the notebook's ``picker`` variable pointing at different objects.
    """

    def __init__(
        self,
        feasible: pd.DataFrame,
        space: dict,
        on_pick: Callable[[dict], object] | None = None,
    ):
        self.on_pick = on_pick
        self.guess: dict | None = None
        self.result = None
        cols = param_columns(feasible)
        self._table = feasible[cols].drop_duplicates().reset_index(drop=True)
        self._space = space
        self._docs = {p["name"]: p["doc"] for p in space["params"]}

        # One control per *parameter*: a permutation parameter spans several
        # pool columns but is picked as one ordering.
        self._groups: dict[str, list[str]] = dict(iter_param_groups(space))
        self._labels: dict[str, dict[tuple, str]] = {}
        for name, dims in self._groups.items():
            if len(dims) > 1:
                self._labels[name] = ordering_labels(space, name)

        self._dropdowns: dict[str, widgets.Dropdown] = {}
        self._updating = False
        rows = []
        for name, dims in self._groups.items():
            dd = widgets.Dropdown(options=[ANY], value=ANY, layout=widgets.Layout(width="11em"))
            dd.observe(self._on_change, names="value")
            self._dropdowns[name] = dd
            doc = self._docs[name].split(";")[0]
            rows.append(
                widgets.HBox(
                    [
                        widgets.Label(name, layout=widgets.Layout(width="9em")),
                        dd,
                        widgets.HTML(f"<span style='color:#52514e'>{doc}</span>"),
                    ]
                )
            )
        self._status = widgets.HTML()
        self._reset = widgets.Button(description="reset", layout=widgets.Layout(width="6em"))
        self._reset.on_click(lambda _: self.reset())
        self._simulate = widgets.Button(
            description="Simulate", button_style="primary", disabled=True,
            layout=widgets.Layout(width="8em"),
        )
        self._simulate.on_click(self._on_simulate)
        self._out = widgets.Output()
        self._box = widgets.VBox(
            rows + [widgets.HBox([self._reset, self._simulate, self._status]), self._out]
        )
        self._refresh()

    def _on_simulate(self, _button):
        self._out.clear_output()
        with self._out:
            if not self.is_complete:
                print("fix every parameter first")
                return
            self.guess = self.config
            if self.on_pick is None:
                print(self.guess)
                return
            self._simulate.disabled = True
            try:
                self.result = self.on_pick(self.guess)
            finally:
                self._simulate.disabled = False
            if self.result is not None:
                display(self.result)

    # -- what the user has fixed so far ------------------------------------

    def _fixed(self) -> dict[str, object]:
        out = {}
        for name, dd in self._dropdowns.items():
            if dd.value != ANY:
                out[name] = dd.value
        return out

    def _mask(self, exclude: str | None = None) -> pd.Series:
        m = pd.Series(True, index=self._table.index)
        for name, value in self._fixed().items():
            if name == exclude:
                continue
            dims = self._groups[name]
            if len(dims) == 1:
                m &= self._table[dims[0]] == value
            else:
                key = next(k for k, lab in self._labels[name].items() if lab == value)
                for d, v in zip(dims, key):
                    m &= self._table[d] == v
        return m

    def _choices(self, name: str) -> list:
        rows = self._table[self._mask(exclude=name)]
        dims = self._groups[name]
        if len(dims) == 1:
            return sorted(rows[dims[0]].unique().tolist())
        keys = {tuple(r) for r in rows[dims].itertuples(index=False)}
        return sorted(self._labels[name][k] for k in keys if k in self._labels[name])

    def _refresh(self):
        self._updating = True
        try:
            for name, dd in self._dropdowns.items():
                choices = self._choices(name)
                current = dd.value
                dd.options = [ANY] + choices
                dd.value = current if current in choices else ANY
        finally:
            self._updating = False
        n = int(self._mask().sum())
        fixed = len(self._fixed())
        self._simulate.disabled = n != 1
        if n == 1:
            self._status.value = "<b>one configuration left</b> — press Simulate"
        else:
            self._status.value = f"{n:,} feasible configurations match ({fixed}/{len(self._groups)} parameters fixed)"

    def _on_change(self, _change):
        if not self._updating:
            self._refresh()

    # -- public ------------------------------------------------------------

    def reset(self):
        self._updating = True
        for dd in self._dropdowns.values():
            dd.value = ANY
        self._updating = False
        self._refresh()

    def show(self):
        display(self._box)
        return self

    @property
    def remaining(self) -> pd.DataFrame:
        """The feasible configurations that match the current choices."""
        return self._table[self._mask()]

    @property
    def is_complete(self) -> bool:
        return len(self.remaining) == 1

    def default_guess(self) -> dict[str, object]:
        """A configuration a newcomer might pick: every tasklet, middling tiles.

        Deterministic, and feasible by construction: it is a row of the
        feasible set -- the middle one among those using all 16 tasklets, by
        the size of the MRAM tile of the reduction dimension.
        """
        rows = self._table
        t = rows["tasklets"].max()
        cands = rows[rows["tasklets"] == t]
        k_cols = [c for c in cands.columns if c.endswith(".mram")]
        cands = cands.sort_values(k_cols)
        row = cands.iloc[len(cands) // 2]
        return {k: (int(v) if isinstance(v, (int, float)) else v) for k, v in row.items()}

    @property
    def config(self) -> dict[str, object]:
        """The chosen configuration as pool-column values, if exactly one is left."""
        rows = self.remaining
        if len(rows) != 1:
            raise ValueError(
                f"{len(rows)} configurations still match; fix more parameters"
            )
        return {k: (int(v) if isinstance(v, (int, float)) else v) for k, v in rows.iloc[0].items()}
