# Context

## Terms

**Project**, **Event** and **Layer** keep their `brustt-platform` meanings.

**Cube**: the local Zarr store holding Tessera embeddings for an area and year,
written block by block and recording the `geotessera` version that produced it.
This differs from Cube in `flood-monitoring`.

**Block**: a fixed-size window of a Cube, the unit that is read, written and
resumed.

**Scale**: the factor that maps a stored quantized value back to its real
embedding value.

**No Data**: pixels with no embedding (outside Tessera coverage or not yet
written); stored as 0 and excluded when decoding.
