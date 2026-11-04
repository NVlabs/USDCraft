# Bezels and Frames

## Purpose

Use `BezelGeometry` for framed openings, trims, and recessed surrounds instead
of building them from stacked rounded boxes and manual cutouts.

## Import

```python
from sdk import (
    BezelGeometry,
    BezelFace,
    BezelRecess,
    BezelVisor,
    BezelFlange,
    BezelMounts,
    BezelCutout,
    BezelEdgeFeature,
    mesh_from_geometry,
)
```

## Recommended APIs

| Shape intent | Helper |
| --- | --- |
| Default framed opening, trim ring, or panel surround | `BezelGeometry` |
| Front-lip and trim-face styling | `BezelFace` |
| Recessed pocket or display/instrument inset | `BezelRecess` |
| Hooded or visor-like top extension | `BezelVisor` |
| Rear flange and mounting extensions | `BezelFlange` |
| Bosses, tabs, and rear mounting patterns | `BezelMounts` |
| Edge notches and local trim interruptions | `BezelCutout`, `BezelEdgeFeature` |

## API Reference

### `BezelGeometry`

```python
BezelGeometry(
    opening_size,
    outer_size,
    depth,
    *,
    opening_shape: Literal["rect", "rounded_rect", "circle", "ellipse", "superellipse"] = "rounded_rect",
    outer_shape: Literal["rect", "rounded_rect", "circle", "ellipse", "superellipse"] = "rounded_rect",
    opening_corner_radius: float = 0.0,
    outer_corner_radius: float = 0.0,
    wall: float | tuple[float, float, float, float] | None = None,
    face: BezelFace | None = None,
    recess: BezelRecess | None = None,
    visor: BezelVisor | None = None,
    flange: BezelFlange | None = None,
    mounts: BezelMounts | None = None,
    cutouts: Sequence[BezelCutout] = (),
    edge_features: Sequence[BezelEdgeFeature] = (),
    center: bool = True,
)
```

- Builds a framed opening with optional recesses, visors, flanges, rear
  mounts, edge cutouts, and trim features.
- The opening lies in local `XY`; depth extends along `Z`.
- `center=False` places the rear mounting face on `z=0`.
- Use this for display frames, instrument trims, light/lens bezels, and
  recessed panel surrounds.

## Detail parameters

Lengths use metres; parameters ending in `_deg` use degrees. Omit optional
detail groups when that feature is absent.

```text
BezelFace(
    style: Literal['flat', 'rounded', 'chamfered', 'radiused_step'] = 'flat',
    front_lip: float = 0.0,
    chamfer: float = 0.0,
    fillet: float = 0.0,
)
```

```text
BezelRecess(
    depth: float,
    inset: float,
    floor_radius: float = 0.0,
    wall_draft_deg: float = 0.0,
)
```

```text
BezelVisor(
    top_extension: float = 0.0,
    side_extension: float = 0.0,
    thickness: float = 0.0,
    draft_deg: float = 0.0,
)
```

```text
BezelFlange(
    width: float = 0.0,
    thickness: float = 0.0,
    offset: float = 0.0,
)
```

```text
BezelMounts(
    style: Literal['none', 'bosses', 'tabs', 'rear_flange'] = 'none',
    hole_count: int = 0,
    hole_diameter: Optional[float] = None,
    boss_diameter: Optional[float] = None,
    setback: float = 0.0,
)
```

```text
BezelCutout(
    edge: Literal['top', 'bottom', 'left', 'right'],
    width: float,
    depth: float,
    offset: float = 0.0,
)
```

```text
BezelEdgeFeature(
    style: Literal['bead', 'step', 'notch'] = 'bead',
    edge: Literal['top', 'bottom', 'left', 'right'] = 'top',
    size: float = 0.0,
    offset: float = 0.0,
    extent: float = 0.0,
)
```

## Advice

- Use `BezelGeometry` when the object should read as a trim or frame around an
  opening, not as a solid box with a hole punched in it.
- Model the front face and rear mounting logic explicitly when they are visible.
- Use asymmetric `wall` values when the real trim is bottom-heavy, visor-heavy,
  or offset around the opening.

## Examples

### Rounded-rect display bezel

```python
display_bezel = BezelGeometry(
    (0.080, 0.050),
    (0.110, 0.080),
    0.012,
    opening_shape="rounded_rect",
    outer_shape="rounded_rect",
    opening_corner_radius=0.006,
    outer_corner_radius=0.010,
)
mesh = mesh_from_geometry(display_bezel, "display_bezel")
```

### Recessed instrument bezel

```python
instrument_bezel = BezelGeometry(
    (0.072, 0.048),
    (0.108, 0.084),
    0.016,
    face=BezelFace(style="radiused_step", front_lip=0.003, fillet=0.002),
    recess=BezelRecess(depth=0.005, inset=0.004, floor_radius=0.0015),
    mounts=BezelMounts(style="bosses", hole_count=4, hole_diameter=0.003, boss_diameter=0.007, setback=0.008),
)
mesh = mesh_from_geometry(instrument_bezel, "instrument_bezel")
```

## See Also

- `40_mesh_geometry.md` for lower-level shell and opening helpers
- For a fuller worked script, see the base SDK asymmetrical panel bezel example.
