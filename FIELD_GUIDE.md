# Capturing scans that stitch well

Short enough to read on your phone at the venue.

## The one rule that matters most

**Every scan should share a good chunk of space with at least one other scan.** Aim for neighbouring scans to
share a third or more of what they see. In practice: walk a few steps and scan again, rather than jumping to the
next room. Generous overlap costs nothing at capture time; thin overlap is expensive to fix afterwards.

## Walk in a loop, not a line

If you scan A, then B, then C in a straight chain and the B-to-C match is wrong, nothing can catch it. If C also
overlaps A, the loop lets the tool see that the matches disagree and settle them together. Walk the perimeter and
finish where you started.

## Doorways: scan the doorway itself

Two rooms scanned separately, one with the door shut and one with it open, are hard to join. Stand in the doorway
and take a scan that sees into both rooms. That scan overlaps both and becomes the bridge. The same goes for stairs,
corridors and any narrow point between spaces.

## Keep individual scans small

Far-away surfaces come back with fewer points per square metre, and the tool works by matching surface detail. A
scan of a 4 x 5 m room can be three times denser than one scan of a 35 x 30 m hall. Several scans walked through a
big space beat one scan of all of it.

## Let it see the floor

The tool finds which way is up from the floor in each scan. A scan pointed up at a sign or a truss, with almost no
floor in it, cannot be levelled and has to rely on the device's own levelling. Include some floor in every scan,
even just a sweep down at the start.

## Let it see a corner

To pin a scan's position from walls, the tool needs two walls that are not parallel. A scan of one flat surface (a
sign, a single wall) gives it nothing to work with. Get a corner in every scan if you can.

## Do not thin the files

Export at full resolution. Thinning changes which surfaces the tool finds, and can move the answer by metres. The
`.pts` format does not record whether a file was thinned, so the tool cannot always tell.

## Capture for drawing, then for joining

Get a clean scan of each area you need to draw, on its own merits. Then add the connecting scans (doorways,
corridors, the loop back) so the tool can join them. That way the scans are useful even if a join needs work by
hand.

## Quick checklist

- [ ] A third or more overlap between neighbouring scans
- [ ] Walk a loop and finish where you started
- [ ] A scan standing in every doorway
- [ ] Several small scans rather than one big one
- [ ] Floor visible in every scan
- [ ] A corner, or two non-parallel walls, in every scan
- [ ] Export full resolution, no thinning
- [ ] Note roughly where you stood for each one; it helps when placing them
