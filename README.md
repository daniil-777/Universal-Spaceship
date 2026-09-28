# Astro Pilot

A spacecraft in Earth orbit that *learns* to fly through an asteroid field with comets streaking through it, live in
the browser. The pilot is a small neural network trained with proximal policy optimisation (PPO) in TensorFlow.js; the
scene is three.js: a textured Earth (NASA Blue Marble day/night/clouds imagery, 2.9 MB in `textures/`, see
`textures/CREDITS.txt`) with a physically based atmosphere, a hero sun with corona and a subtle lens flare, Jupiter and
Saturn small in the distance, Venus, the Milky Way — and procedural asteroids, comets and the ship. Everything else is
generated in code; the policy file is 360 KB.

Open `index.html` over HTTP (`python3 -m http.server` in this folder) — ES modules do not run from `file://`.

## What you see

* **Colour / Mono / Ink** — the same scene in full colour, in greyscale, or as a strict black-and-white ordered-dither
  print (toggle *Ink on white* for black on white). `M` cycles the modes, `H` hides the interface.
* **Playbox** — pause, reset, simulation speed, asteroid count and belt speed, comet count and comet speed, three
  cameras (Chase from behind the ship — the default — with rocks and comets rushing toward you; Side, the classic
  west-to-east scroll; Orbit — drag to orbit), the flight trail, the ship's sensor beams, and *Fly it yourself* (`W S`
  pitch, `A D` yaw, `Q E` roll, `⇧` boost, `X` brake) to compare your flying with the autopilot's.
* **Training** — press *Train* and 128 ships start learning in a background worker while the page keeps rendering.
  The panel shows steps, episodes, updates, steps/s, the curriculum level, and live charts of the episode return,
  episode length (exploring and deterministic check-ups), collisions per 1k steps, policy and value loss, entropy,
  approximate KL, clip fraction and explained variance, plus a top view of the whole fleet. *New policy* starts from
  random weights so you can watch the first minutes of learning; *Save* / *Load* export or import a policy JSON.
* **HUD** — what the displayed ship sees (its 55 forward sensor beams as a heat grid), what the policy outputs
  (pitch, yaw, roll, thrust), the critic's value estimate, speed, laps and crashes.
* **The spacecraft flies with its surfaces** — four elevons (inboard and outboard on each wing, turning on rounded
  hinge noses inside dark coves), split rudders on both canted fins that also open as a Shuttle-style speed brake when
  braking, a body flap under the engine face, exhaust plumes that vector up to 6°, and vapour streaming off the wingtips
  in hard pulls in the air. They follow the pilot's rate commands plus the measured rotation (so they lead a
  manoeuvre, settle, and check on release) through 60 ms actuators with a 90°/s rate limit; the RCS thrusters still puff
  for attitude changes in space. `samples/ship-control-surfaces.png`, `ship-roll-sequence.png`.
* **Flight board** (lower right; Playbox switch *Flight board*, `?board=0` hides it) — an attitude indicator whose
  aircraft symbol is the ship seen from behind (pitch ladder, bank scale and pointer), a heading tape, speed in km/h
  with Mach in the air, altitude (metres over the ground in the air; the orbit's real altitude from the globe's
  geometry in space), radar altitude, vertical speed, the nearest obstacle, throttle and a warning chip (*PROXIMITY*,
  *PULL UP*, *CRASH*) — beside a live onboard camera: *NOSE* (banks with the ship), *BELLY* (down and ahead) or
  *TOP* (a moving map); click it to switch. The camera is a second render at 30 Hz (15 when the frame rate sags) into
  its own target, composited with the page's tone mapping and colour mode. Scale: 19 m per unit (the Burj Khalifa is
  44 units tall in the procedural Dubai).

**Low passes.** Every ~45 s the ship dives toward the Earth for ~28 s: the planet swells and tilts toward its day side
(continents, mountains, lakes and oceans from the imagery, clouds, the terminator, city lights at night), airliners
with contrails cross the surface below, and space stations — Mir-class Soviet stations, procedural down to the docking
node, the radial modules with their solar arrays, the docked Soyuz, the Sofora girder and the dishes, four draw calls
for all of them — tumble slowly through the corridor as a third kind of hazard the pilot has to avoid. Then the ship climbs back to orbit. The
Playbox switch *Low passes* turns the cycle off. Between passes the ship is on a real orbit (`src/orbit.js`, `src/skyorbit.js`): ISS-like, 420 km up and inclined
51.6°, at 7.66 km/s — 27,600 km/h, one orbit in 92.8 minutes (the board shows it, with the ground speed) — over an Earth
that turns on its axis at the real rate for the real date and time (UTC): `src/ephem.js` computes sidereal time and the
Sun and the Moon with Meeus's algorithms, so the terminator, the day side and the city lights are where they are right
now, and the ground track slides west each orbit as the planet turns beneath it. On the orbit's night side the ship
flies in the Earth's shadow. The **Moon** hangs in the sky at its true direction, size and phase — NASA's LRO colour
mosaic with LOLA relief (`textures/moon/`, the Scientific Visualization Studio's CGI Moon Kit), lit by the Lommel–Seeliger
law of lunar soil (a crisp terminator, the opposition surge at full Moon) with earthshine on its dark side
(`src/moon.js`). Whenever it is lit enough to see (≥ 15 %), the flight starts with the Moon rising over the Earth's
limb beside the ship — over the day side while it waxes and through full, before sunrise in the week it wanes (it then
rises ahead of the Sun) — and it climbs through the frame in about seven minutes, returning once an orbit (`?start=dawn`
keeps the classic dawn pass, `?start=moon` forces the Moon). Its true size is only 0.52° — about ten pixels — so, like
Stellarium's *enlarge Moon*, it is drawn 4× larger by default (Playbox *Moon size*, `?moonscale=1` for the true size);
the *Moon* camera widens by as much, so the telescope's view stays the true one. The *Time warp* slider (1×–300×) runs the whole sky faster. Choosing an atmospheric route re-phases the
orbit so the descent begins over that place (and the low-pass airliners are re-routed through whatever point is under
the ship when a pass begins).

**Lunar orbit** (Playbox *Orbit: Moon*, or `?orbit=moon`, height `?moonalt=` 500–60,000 km, default 1,500): the ship
circles the Moon on a polar orbit (1.23 km/s at 1,500 km, one revolution in 4.6 hours), the Moon filling the lower sky
at its true size with NASA's 4K colour mosaic and LOLA relief (fetched only in this mode), the Earth a small real-size
globe at its true direction and phase (`src/lunarsky.js`). The orbit's plane and starting point are chosen for the
view on the day: the ground ahead in sunlight, the Sun out of the frame, and the Earth in the frame whenever it stands
well away from the Sun (near full Moon it is a thin crescent beside the Sun, behind the ship). The *Moon* camera turns
into a 3.2° telescope on the Earth. Picking a route flies back to the Earth.

**Atmospheric flight** (Playbox switch, or `?atmo=1`): the ship descends to about half an airliner's altitude in this
scene's exaggerated scale and flies a bird's-eye route over **real satellite imagery** streamed as Web-Mercator tiles
(Esri World Imagery; a coarse zoom-9 strip plus a sharp zoom-11 strip under the flight path, each strip one curved
mesh with one texture atlas into which tiles are copied on the GPU as they arrive, scrolled by texture offset so the
surface is seamless — no tile borders, no loading holes; the ground scrolls east along a parallel you choose with the **Route** control, and each
route brings its own world: *Alps* (46.5° N, the alpine range as obstacles), *China* (29.3° N, the Avatar valley of
Zhangjiajie sandstone formations), *New York*, *London*, *Moscow* and *Dubai* (each city's skyline rises from its own imagery as soon as you press
it; switch *Skyline flight* off to see the imagery alone with airliners and birds); a
third zoom-12 strip right under the flight path shows the cities from the top. The strips fade into each other and
into the haze at their edges, open-sea tones are evened out in the shader, a blue sky dome replaces space, aerial haze
fades the distance, and the hazards become **airliners**: no rocks or comets, but six A320-class Airbus jets with
contrails (`src/aircraft.js`: a lofted fuselage with the drooped nose, lofted wings with sharklets, engines with intake
lips and fan faces, a drawn livery with cabin windows, cockpit panes and a blue fin — 4.6 k triangles, two draw calls
for all of them) — head-on,
crossing or overtaken — that the pilot has to avoid (it treats them like slow rocks). Switch it off to climb back to
orbit; the credit line names the imagery source while it is on.

**Mountains — a third thing to learn.** On the *Alps* route a procedural mountain range (`src/heightfield.js`:
domain-warped ridged multifractal noise, periodic along the corridor, every world's highest summit exactly 40 units
above the valley floor, i.e. well *above* the ship's cruising altitude) rises from the imagery. It is part of the
*environment*, not just the picture: the same height field is sampled by the 55 sensor beams (ray marching), by the
ground-clearance proximity penalty and by the collision test, and over the mountains the air thins out — a soft ceiling
(y = 12) stops the pilot from simply climbing above everything — so it has to thread the passes and skirt the massifs
while dodging airliners. Flying straight at the spawn altitude hits a mountain about 8 times a minute. The renderer
(`src/mountains.js`) draws it as real Alps: a 0.25-unit mesh with bedding ledges folded into the rock, one batched draw
call with distance levels, and a per-pixel material — conifer forest with canopy shadows, alpine meadow, bedded gneiss
in greys, browns and rusty high beds, couloirs and erosion streaks running down the faces, scree fans under the cliffs,
snow that settles by altitude, slope, curvature and lee side and pokes the rock through — lit by a low golden sun with
cast shadows from the shadow field, drifting cloud shadows, sky light with baked ambient occlusion, bounce, rim light,
and valley mist pooling below; the chase camera is kept out of the rock. `samples/4k/alps-4k.png` is a 4K render.
The *China* route is the **Avatar valley** — Zhangjiajie's sandstone pillars (the "Avatar mountains") as a whole
landscape. Ten Meshy AI models: eight Meshy 7 Ultra formations generated from prompt-made reference images (a colossal
pillar, a needle cluster, a natural stone arch, a waterfall massif, a Tianzi ridge wall, vine-hung Hallelujah pillars, a
pagoda summit and Guilin karst domes, each rising from its own forested mound of roots and boulders), plus the two
clusters made from a photograph of Wulingyuan. About 900 of them stand along a 960-unit valley that repeats only after
eight corridor laps (`src/avatarlayout.js`): a river winds down the middle as the flight lane, formations line both
banks and stand on five islands, three staggered canyon walls rise beyond them, and massifs of two to six formations
stacked together fill the land out to the horizon, growing into giants with distance. The mountains **grow out of the
ground** rather than stand on it: the valley floor (`src/valleyground.js`) is real terrain at the rock's scale —
rolling forest canopy, a flood plain, gravel banks, flowing jade water — that rises into a foothill around every
formation's footprint, so each rock's rim sits below its own hill and its jungle skirt runs into the forest floor, with
scree and contact shade at the foot and soft sun shadows from the shadow field; near the ship thousands of instanced
tree crowns (`src/valleytrees.js`) give the forest volume. There is no satellite photograph under the valley: its
kilometre-scale fields made the rock read as if it hovered high above a distant landscape. Each formation has four
levels of detail (220k / 110k / 25k / 5k triangles, 400k near for the tunnel rocks, chosen by the distance to its
footprint) and is culled to the view. All four are re-baked from Meshy's 7.6-million-triangle originals by the harness's
`lod_bake.mjs`: the shape is simplified freely, gets a fresh texture atlas, and every texel is painted from the closest
point on the original — the base colour, and on the near and mid levels also a normal map (the original's fine relief)
and a roughness map (the city tiles' extra-far levels are re-baked the same way). Simplifying across Meshy's own UV
layout (some 900,000 islands of one or two triangles) had dragged the texture into zigzag streaks and speckle and flipped
a third of the mid levels' triangles, which then showed as holes. Detail streams ahead of the ship: finer levels load well before the camera needs them, the
valley and the single cities then preload every remaining level in the background and keep them, and textures go to
the GPU when a level arrives instead of on the frame it is first drawn. The ground and the formations near the corridor are baked into the collision field
(`src/avatar_grid.js`, 1920 × 180, fetched with the route), so the pilot dodges the very rock it sees, and the lane
stays at least 24 units wide. The flight board names the zone below: Hallelujah Peaks, Tianzi Ridge, Golden Whip
Stream, Karst Gardens. The first twelve-cluster pillar world (`src/china_grid.js`) remains in the training mix, and
stands in on the China route if the valley's map cannot load. `samples/avatar-valley.png`, `avatar-valley-river.png`
and `samples/4k/avatar-valley-4k.png` show it.

Three **tunnel rocks** close the lane once per lap each: a Heaven's Gate massif with a round cave high through it, a
jungle cave with the river running out of it, and a gate with a waterfall beside its mouth (Meshy 7 Ultra, turned so
the bore runs along the flight path, their depth squeezed so the tunnels are 20 to 55 units long). The collision field
has a second layer for them: over each bore, the tunnel's floor and its roof, baked from the bore's measured profile
(the open window rays pass all the way through), so the ship collides with the roof and walls it sees and not with the
air in between; the beams see the roof too. The pilots never learned to thread a tunnel, so a **tunnel guide** in the
fly-by-wire takes over from 80 units before an entrance, fully from 35: it pursues a point on the bore's centre line,
keeps the wings level and the speed at cruise, and hands back after the exit. The chase camera drops to a low, level
view that frames the cave mouth and stays under the roof; bird flocks keep out of the tunnels; when you fly yourself
the guide is off. In evaluation the autopilot threaded 107 of 107 tunnels.

**Waterfalls** pour from 37 ledges on the formations beside the corridor and on a few giants far out (the ledges found
on each model by the harness's `falls_probe.mjs`: a sheer face below, no overhang above), and two fall across the
Heaven's Gate and the waterfall gate's mouths, so the ship flies past the water into the rock (`src/valleyfalls.js`):
instanced curtains that arc off the lip and widen as they fall, streaks racing down faster lower, a veil of spray
around each, and mist churning where they land.
The city routes (New York, London, Dubai, Moscow) are **joined Meshy districts**: square city-block dioramas, each one Meshy 7 Ultra image-to-3D model
of a prompt-generated reference (streets along all four edges, the same 45° framing), joined edge to edge across the
corridor — New York's Lower Manhattan (One World Trade Center), Midtown East (the Chrysler) and Hudson Yards on
40-unit tiles with last round's photo models (Midtown from the Hudson, the Empire State, a pencil tower) on both
flanks; Dubai's Downtown (the Burj Khalifa), Sheikh Zayed Road and Business Bay, and Moscow's Moscow City, a
Stalinist high-rise and Red Square on 60-unit tiles. `src/meshyworld.js` fits each tile by its plate (a precise
bounding box; Downtown's plate sits turned 45° on a sand plinth), sinks it until the streets along its edges meet the
ground, trims its slab rim per instance, lays an asphalt base under the rows, and bends every vertex with the shared
Earth curvature. Each model has three levels of detail (150k triangles near, a seam-preserving 25–60k far, a sloppy
~9k extra-far; `test/tiles_build.mjs` makes them from the raw ~100 MB Meshy files): the coarse levels load first, a
model's near level only when one of its placements comes close, and idle near levels are dropped. The near level keeps
220k triangles, a sharpened 2K base colour, a 1K normal map and Meshy's roughness map. Shading makes them read as real
places, not clay: sun shadows cast by the towers onto the streets and each other (the collision height field on the
GPU, marched up-sun), street-level and street-canyon occlusion, and glass that reflects a daylight sky where Meshy's
roughness says smooth (metalness stays 0 — Meshy's metalness maps turned whole towers into sky mirrors). In atmospheric
flight every PBR surface — glass, the ship's hull, airliners, the Eiffel Towers — reflects a baked daylight sky with a
sun instead of space. The tiles are baked into the collision field (`src/<city>_grid.js`), so the pilot dodges the
buildings it sees.

The **Megacity** route merges everything into one map 420 units across and 960 long — eight corridor laps, over a
minute before it repeats: New York, London (Westminster with Big Ben, the City with the Gherkin, the Shard quarter),
Moscow and Dubai zones in a loop, each border bridged by a fusion district that mixes the two cities' architecture,
and a Paris plaza per zone. Variety beyond the 17 generated districts comes from Meshy retextures (the same geometry
re-skinned in another style: bronze Hudson Yards, brick Midtown, gold Moscow City, a red-brick Stalinist tower, marble
Sheikh Zayed Road…), remixes (two districts cut along a street each — found by `test/cuts_meshy.mjs` — and joined,
the second slid so the streets meet; the renderer and the bake clip each half per instance) and skyline stretches.
The environment slides the corridor 120 units along the map every lap (`hf.lap.shift`), the renderer and the shadow
field follow, and the 460 KB collision grid (`src/mega_grid.js`) is fetched only when the route is chosen.
Each zone's Paris plaza (a Meshy Haussmann block around a lawn) holds an **Eiffel Tower** in the zone's style —
chrome Art Deco for New York, the classic brown for London, constructivist red with a ruby star for Moscow, slender
ivory and gold for Dubai — and two 88-unit giants (gold with light strings, a twisted white-and-teal futurist) stand
on outer plazas over the skyline. `src/eiffel.js` builds them in the browser as real lattice: four curved legs
following the tower's exponential profile, box-girder chords with diagonal bracing, the great arches, three
platforms, the campanile and antenna, light strings that bloom — ~200k triangles near, ~25k far, ~2k extra-far, in
30–50 ms per tower.
Training uses mixed worlds (half the parallel environments are asteroid belts, a quarter mountains with airliners, a
quarter city skylines) so one policy handles every regime.

**Skyline flight — the fourth thing to learn.** The *Skyline flight* switch (or `?skyline=1`) takes the ship down among
the skyscrapers of the route's city: *Dubai* (Sheikh Zayed Road lined with glass towers, the Marina and JLT clusters,
Downtown with the Burj Khalifa rising into the thin air), *New York* (the Manhattan grid, Central Park, the Midtown
super-talls, the Empire State and Chrysler setbacks, One World Trade Center downtown), *London* (the Thames bending
along the corridor, Westminster with Big Ben, St Paul's, the City cluster — 22 Bishopsgate, the Gherkin, the
Cheesegrater, the Walkie-Talkie — the Shard, Tower Bridge with its deck across the river, Canary Wharf, Battersea Power
Station, brick terraces everywhere) and *Moscow* (rows of Soviet slabs, the University, the Kremlin and the old centre,
the Seven Sisters, the Moscow City cluster). A city route turns the skyline on by itself. The towers are built from axis-aligned footprints — boxes, round and octagonal
towers, tapered towers, setback stacks (`src/cityfield.js`: landmarks placed by hand, filler blocks from a seeded
generator) — and rasterised shape by shape into the same height field as the mountains, so
the beams, the collision test (which also checks the ship's flanks, so a wingtip clipping a wall counts) and the ground
clearance all work unchanged; the ceiling applies, so the pilot threads the canyons rather than climbing over them.
Straight flight hits a tower every 5–30 s depending on the city. Everything is procedural and drawn in a few instanced
draw calls per city, one per shape (`src/city.js`). The facades are real drawn textures generated at load, no files
(`src/facades.js`: eight 512² tiles in one texture array — three curtain-wall styles with frames, spandrels and blinds,
premium floor-to-ceiling glass, running-bond brick with mortar and punched windows with sills and lintels, ashlar stone,
precast panels with balconies, raw concrete, rain streaks on all of them); the glass mask in their alpha channel tells
the shader where to add the reflection of sky, horizon, ground and sun. On top: cornices on stone and brick, dark glass
lobby plinths, rooftop plant, masts and helipads, canyon shading from the height field, real cast shadows.
`samples/4k/` holds 3840 × 2160 renders of New York, Dubai and Moscow, the Megacity's London and Moscow zones with their Eiffel Towers, the classic tower on its own, the Alps and the Avatar valley. Shadows are real
but cheap: the active height field lives on the GPU as one small float texture (`src/shadowfield.js`) and every facade
and ground pixel marches ten samples up-sun through it — towers shade each other and the streets, ridges shade their
valleys — with no shadow maps or extra passes. The chase camera pulls in when a tower comes between it and the ship.

**Birds — only in the atmosphere.** Over the mountains and the cities, flocks of birds wander through the flight band:
large birds in V formations and small birds in swirling swarms (`src/birds.js`: instanced silhouettes that flap, glide
and bank in the vertex shader — one draw call for every bird). A flock is a hazard like any other (a bird strike ends
the episode): it wanders with a random-walk heading, keeps above whatever stands below and turns back at the corridor
sides, and the pilot sees it with the same beams and time-to-contact ranking as rocks and airliners. There are no birds
in orbit.

**Landing at the airport (`?scenario=landing`, or *Land at the airport* in the Playbox).** A separate scenario in real
units: the ship at full scale — a 36-m delta-wing spaceplane, 72 t, landing gear with oleo legs, ground spoilers and a
drag chute (`src/landing/vehicle.js`, `flight.js`) — flies a complete instrument approach to Astro Pilot Spaceport's
4-km runway 26 and lands itself. The airport is laid out to ICAO Annex 14 (`airport.js`): threshold, aiming-point,
touchdown-zone and centreline paint, edge/centreline/touchdown-zone lights with the end colour coding, an ALSF-2-style
approach light system with its sequenced flashers, a PAPI whose red and white come from the viewer's actual angle, an
ILS, taxiways with rapid exits and their signs (distance-remaining boards, exit and hold-position signs), an apron
with floodlights, terminal, tower and hangars. The weather is a METAR (`wind=23015G25`,
`turb=light|moderate|severe`) on a boundary-layer wind profile with MIL-F-8785C Dryden turbulence, and a cloud
deck that follows the visibility (`vis=cavok|haze|fog`: a few clouds high up, a scattered layer at 3000 ft, or an
overcast at 300 ft that the ship descends through to break out over the approach lights; `clouds=BKN&ceiling=1200`
sets another), by day, at dusk or at night (`time=`).

The autoland (`autoland.js`) is the way certified CAT III systems fly, tuned by AI: a Dubins path from wherever the ship
is to where ATC would vector it onto the localizer (long enough to lose the height at ≤ 3.5° — a longer final, or a
360, only when it must), L1 lateral guidance, localizer and glideslope capture, a TECS energy law on the thrust
with the path on the pitch, stabilized-approach gates (else a go-around), an exponential flare whose touchdown sink is
re-solved five times a second so the touchdown lands on the aim point (model-predictive control), a partial de-crab with
a wing-low bank, retard, derotation, autobrake and chute, and a brake-to-vacate rollout onto a rapid-exit taxiway. Its
two dozen gains and thresholds were chosen by CMA-ES over Monte Carlo campaigns of thousands of landings in drawn
weather (headwind −10…25 kt, crosswind ±20 kt, gusts, turbulence) — on 2,400 unseen landings (1,500 from anywhere 16–30 km out, 900 vectored finals) it landed every one, was
stabilized by 1000 ft in 99.9 %, and touched down 393–564 m past the threshold (median 456 m) at a median sink rate of
2.3 ft/s — what airline flight-data monitoring finds typical — within 5.4 m of the centreline, at ≤ 3° of bank and ≤ 1.6 g. Cameras: cinematic, chase, a
cockpit HUD, the tower, the runway side and under the approach; the sound is procedural WebAudio (the engines on their
spool, the wind, the tyres' squeal and thump by the sink rate, the rollout's rumble) with spoken callouts and the
tower's clearance and exit calls; each landing ends with a report card. It runs at 60 fps at 1600×900 in every
weather and light (`samples/landing-approach-farmland.jpg`, `landing-fog-approach-lights-hud.jpg`, `landing-dusk-final.jpg`,
`landing-night-flare.jpg`, `landing-rollout-chute.jpg`).

The ship always flies west → east through a corridor that is a torus in x: in the Side view it leaves the screen on
the right and re-enters on the left (drawn twice near the seam so the wrap is seamless); in the Chase view the wrap is
invisible because everything is drawn relative to the ship. The backdrop swings around with the camera mode so the
planets sit ahead of the ship in the chase view.

## The Earth zoom

**Zoom in** in the Playbox (or **Z**) opens a telescope view of the real Earth under the ship: drag to pan, wheel or pinch
to zoom from 20,000 km down to ~300 m above the ground, right-drag (or Ctrl-drag) to tilt and turn, WASD / arrows / + / −
/ Q / E / R / F from the keyboard, **Z** or **Esc** to come back. The ship keeps flying underneath (the learned pilot takes
over while you look around) and a blue dot marks where it is. The imagery is open data fetched live, sharper as you go
down: NASA Blue Marble (≈ 600 m per pixel), Sentinel-2 cloudless by EOX (≈ 10 m), and Esri World Imagery for the closest
levels (under 1 m in many places); real relief comes from AWS Terrain Tiles (≈ 30 m). It is drawn as five nested rings of
tiles around the point you look at (`src/earthrings.js`), each ring a curved mesh with one GPU atlas like the terrain
strips and each one level coarser than the ring inside it; a ring shows once its tiles are in, so a coarser one — or the
Blue Marble globe with its procedural relief — covers whatever is still loading, and a coarse ring leaves a hole where
the finer one is in (otherwise its smoothed valleys would hide the sharp ones). With no network the view still works,
just without the sharper imagery. **Always day** lights the view as if it were midday; otherwise the real Sun of the sky
clock lights it. Known limitation: very low and steeply tilted over cliffs, the draped photos smear on near-vertical faces.

## Files

| file | purpose |
|---|---|
| `index.html` | page shell, styling, import map (three.js 0.170 and TensorFlow.js 4.22 from jsDelivr) |
| `src/env.js` | the flight environment: physics, asteroid model, sensors, reward, curriculum (pure JS, shared by training and display) |
| `src/ppo.js`, `src/actor.js` | PPO agent on TF.js; inference runs on a plain-JS mirror of the weights |
| `src/trainer.js` | 128 environments in lock-step, GAE, curriculum, deterministic evaluation |
| `src/train-worker.js`, `src/train-client.js` | training in a module Web Worker (main-thread fallback) |
| `src/space.js`, `src/earth.js` | the backdrop: Earth (day/night blend, normal-mapped relief, ocean glint, cloud deck with shadows, single-scattering atmosphere), sun + flare, far planets, stars |
| `src/asteroids.js`, `src/comets.js`, `src/satellites.js`, `src/aircraft.js`, `src/planes.js`, `src/terrain.js`, `src/ship.js` | the procedural asteroid field, the comets (particle dust tail, filamentary ion tail, coma), the Soviet space stations, airliners over the globe (low passes) and in the corridor (atmospheric flight), the streamed satellite-imagery terrain, the spacecraft |
| `textures/` | NASA Blue Marble Earth imagery via the three.js repository (credits inside) |
| `src/app.js`, `src/camera.js`, `src/post.js`, `src/ui.js`, `src/charts.js` | scene glue, cameras, mono/ink pass, panels, charts |
| `src/meshylayout.js`, `src/meshyworld.js`, `src/*_grid.js` | the Meshy models (Zhangjiajie pillars; the district tiles of New York, London, Moscow, Dubai, Paris and the fusion districts; last round's photo models): layouts incl. the megacity generator, the instanced multi-level renderer, baked collision grids |
| `src/avatarlayout.js`, `src/valleyground.js` | the Avatar valley: its formation catalogue (tunnel bores, waterfall ledges) and seeded layout (river, banks, islands, canyon walls, stacked massifs, tunnel rocks, falls), the ground height with a foothill under every formation; the valley floor and river renderer |
| `src/edgefence.js` | the corridor's invisible edges made visible where the ship meets one (a faint grid on the edge ahead of the nose) |
| `src/valleytrees.js`, `src/valleyfalls.js` | the valley's tree crowns near the camera; its animated waterfalls and their mist |
| `src/board.js`, `src/pipcam.js` | the flight board (attitude indicator, heading tape, readouts) and its onboard camera |
| `src/eiffel.js` | the procedural Eiffel Towers (six variants, three levels of detail) |
| `src/ship.js`, `src/shipsurfaces.js` | the spacecraft and its moving control surfaces (elevons, split rudders / speed brake, body flap, vectored plumes, wingtip vapour) |
| `src/ephem.js`, `src/orbit.js`, `src/skyorbit.js`, `src/moon.js`, `textures/moon/` | the real sky: UTC clock with time warp, sidereal time, Sun and Moon (Meeus), the 420-km orbit with J2 drift, the sky in the ship's frame, the Moon (NASA SVS CGI Moon Kit) |
| `src/earthtiles.js`, `src/earthrings.js`, `src/earthzoom.js` | the Earth zoom: tile, level and view maths (pure JS, tested); the nested imagery rings with Terrarium relief and their tile loader; the telescope view (scene, controls, HUD) |
| `src/lunarsky.js` | the lunar orbit's sky (`?orbit=moon`): the polar orbit round the Moon, the Earth, the Sun and the view-composed start |
| `src/landing/*.js` | the landing scenario (`?scenario=landing`): SI flight model and gear, METAR wind and Dryden turbulence, the ICAO airport as data and in 3D (paint, lights, PAPI, ILS, signs, buildings), cloud decks, the Dubins/L1/TECS autoland with its predictive flare and rollout, Monte Carlo sim, scene, HUD, tower calls and sound |
| `model/policy.json`, `model/policy_atmo.json` | the pretrained pilots (weights + observation normaliser): the belt pilot, and the atmospheric pilot fetched when atmospheric flight is first switched on |
| `tests/*.test.mjs` | `node --test tests/*.test.mjs` — physics, sensors, GAE, log-probs, TF/JS mirror equality, export round trip |

## The asteroid model

Shapes are icospheres displaced by fractional Brownian noise (`ρ = 1 + 0.32·fbm(2.1 n̂) + …`), scaled by a random
ellipsoid, carved with 3–8 craters (smooth-step bowls with a raised rim) and finer pits; some are contact binaries
(two blended lobes, like Itokawa), some spinning tops with an equatorial ridge (like Bennu). Vertex colours mix grey
and umber, darkened in cavities. Twelve shapes are instanced. Each rock keeps one of the real albedo classes as an
instance tint — dark carbonaceous, warm stony, bright metallic, reddish primitive, grey — and the shading keeps an
airless body's hard terminator: the camera-side fill light keeps the night side readable without lighting it.

Sizes follow the collisional-cascade power law `N(>r) ∝ r^-2.3` (`r = r_min (1−U)^(−1/2.3)`, clipped to 1–5).
Motion: every rock has a mean westward drift (2–7 units/s, the belt streams past the ship) with a Keplerian-like shear
across depth (`v_x += 0.06 z`), an Ornstein–Uhlenbeck random component (`dv = θ(v̄ − v)dt + σ√dt·N(0,1)`, θ = 0.4 s⁻¹,
σ = 1.2: smooth, mean-reverting wander rather than jitter), free rotation about a fixed axis (bigger rocks spin slower),
elastic sphere collisions between rocks (mass ∝ r³), periodic wrap in x and reflection at the corridor walls.

**Comets** are a second population: a small icy nucleus (collision radius 1–1.8) that enters through one face of the
corridor on a straight line at 10–20 units/s, crosses, leaves and respawns elsewhere — never on top of the ship. They
live in the same hazard list, so the sensors, the time-to-contact ranking and the collision test treat them like any
rock (only faster). Visually (after real comets such as Hale–Bopp and NEOWISE): a compact coma compressed on the
sunward side, a broad curved dust tail of ~480 particles per comet born at the nucleus and pushed anti-sunward by
radiation pressure (integrated in the vertex shader), and a delicate blue ion tail of three filaments with flowing
noise, kinks and an occasional disconnection knot — four draw calls for all comets together.

## The pilot

* **Observation (115 numbers, body frame):** 55 cone-shaped sensor beams (11 azimuths × 5 elevations, ±75° × ±40°,
  range 50), the six most urgent rocks ranked by time-to-contact (relative position, velocity, radius), the ship's own
  velocity, rates, attitude vectors, position across the corridor and its filtered command.
* **Action (4 numbers, 15 Hz):** roll, yaw and pitch *rate* commands and thrust. A fly-by-wire controller tracks the
  rates (0.15 s) and pulls the velocity toward cruise speed along the nose, so the ship flies where it points and the
  motion is smooth by construction. Boosting is possible but costs reward.
* **Edges:** the corridor is 36 units wide inside a far wider world, so its sides are invisible walls (in space also its
  ceiling and floor). The beams see them — in the sensor display a beam that meets an edge turns amber, and a faint grid
  lights up on the edge where the nose would cross it — and an **edge guard** in the fly-by-wire turns the ship away
  when it would reach one within 1.7 s: a rotation of the nose back toward the corridor, banking into the turn at a side
  wall, blended from no authority at 1.7 s to full authority at 0.6 s (the velocity trails the nose by ~0.7 s, so it
  has to start early). The flight board shows EDGE while it steers (`samples/edge-guard.png`: the ship banking away from
  the valley's side edge, sensor beams on). The pilots' own inputs are unchanged (they know
  where the walls are from their position across the corridor); both were fine-tuned with the guard in the loop.
* **Reward:** progress east (capped at cruise speed), nose east and wings level, small penalties for angular rate,
  command changes and effort, a proximity potential around the nearest rocks, soft walls, and −20 on impact.
* **Learning:** PPO-Clip with separate actor/critic MLPs (115→128→128, tanh, orthogonal init), GAE(0.99, 0.95),
  observation and return normalisation, value clipping, entropy bonus, global-norm gradient clipping, KL-adaptive
  learning rate, KL early stopping, proper time-limit bootstrapping, and an automatic curriculum (10 → 40 rocks,
  faster and bigger, advanced by deterministic check-ups). 128 environments × 64 steps per update, minibatches of
  2048 — TF.js is bound by per-operation overhead, so few large minibatches are what make in-browser training fast
  (≈ 6k environment steps per second on an M3 Pro).

Two pretrained pilots fly the page. The belt pilot (`model/policy.json`, run 24): 47.9 M environment steps of
headless browser training, several runs chained (torque-command runs failed to learn dodging at all — a zero-action
baseline survived as long — so the action space became rate commands; later runs added soft sensor beams, then mixed
asteroid / mountain / city worlds, bird flocks, the Zhangjiajie pillars, and last 8 M steps with the edge guard). The
atmospheric pilot (`model/policy_atmo.json`, run 20, 59 M steps): continued in city-heavy mixes as the Meshy
districts, the joined tile cities and the megacity arrived; it takes over whenever atmospheric flight is on, because
every continuation traded belt skill for atmospheric skill and one policy could not hold both (and further
continuations of it, with or without the guard in the loop, only lost skill). Deterministic evaluation with the edge
guard on, crashes per minute (48 episodes per belt level, 40 per atmospheric world; "straight" = flying dead ahead, no
policy): belt pilot — asteroid belt 25 + 2 comets 0.9 (straight 1.5), 31 + 3 → 1.3 (straight 9.6), 40 + 4 → 1.8
(straight 5.3); atmospheric pilot — the Avatar valley 0.3; Alps with 6 airliners and 3 flocks 2.6 (straight 14.2 — the
densest world); the first Zhangjiajie clusters 2.1 (straight 16.2); New York 0.6 (straight 2.1); Dubai 0.3 (straight
1.5); Moscow 0.6 (straight 4.9); London 0.8 (straight 6.4); the Megacity 1.5 (straight 2.8) — on average the same as
without the guard (1.22 vs 1.23 over the seven older worlds), while the time spent scraping a side edge fell from up to
2 % to at most 0.3 % (the Alps, where the terrain is lowest along the edges, from 13 % to 7 %). They cruise at 13–14 ± 4
units/s (2.5–22), braking or boosting most of the
time, mean turn rate 0.5–0.6 rad/s, and slow down among the peaks and pillars. The Training panel shows the belt run's
curves on load; press *Train* to keep improving the belt pilot in your browser (the curves continue; while you train,
it flies everywhere), or *New policy* to watch learning from scratch.

## Embedding in a web page and performance

Copy the folder (`index.html`, `src/`, `model/`, `textures/`; `samples/`, `tests/` and `README.md` are optional) and
embed it as an `<iframe src="astro-pilot/index.html" style="width:100%;aspect-ratio:16/9;border:0"></iframe>` or link
to it. Weight on the wire: ~1.6 MB of Earth textures (2K), two ~190 KB pilots (float16 weights + training curves; the atmospheric one is fetched on first descent), ~420 KB
of ES-module source (no build step; the procedural cities, mountains and their shadows add no textures at all), plus
Meshy models fetched only on their route, coarse levels first (the Avatar valley 6 MB up front — its distant levels and
its 0.46 MB map — then 17 MB of middle detail and up to 30 MB of near detail streamed as you fly; New York 5.7 + 13.2 MB; Dubai 4.1 + 11.2 MB; Moscow 2.4 + 6.7 MB; the Megacity 21 MB up front and its near
levels, up to 55 MB, streamed and evicted along the flight), plus three.js from jsDelivr (~600 KB gzipped, cached across sites); TensorFlow.js
(~1.3 MB) is fetched only when *Train* is pressed; satellite tiles only in atmospheric flight. Runtime: the pixel ratio
is capped at 1.5×, SMAA is skipped above 1.4×, and an adaptive render scale lowers the resolution (down to 0.6×) when
the frame rate drops below 40 fps and restores it when there is headroom; the four high-detail asteroid shapes are used
only for big rocks. On an M3 Pro the page holds 60 fps in every mode; the heaviest scene (atmospheric flight with the
three imagery strips, mountains and jets) draws ~35 draw calls; a skyline adds up to six. The imagery strips, the range and
the towers all bend around one shared Earth curvature, so nothing floats above the ground at a distance.

## Retraining

`/Volumes/LaCie/astro-pilot/test/` holds a Playwright harness: `node page_check.mjs` (smoke test with screenshots),
`node train_headless.mjs --page /__pages/train.html --steps 4000000 --nenvs 128 --name run [--init /__pages/x.json
--level 0.7]` (headless training on the GPU; writes `runs/run/policy.json`, `metrics.json`, `eval.json` and a
`live.json` every 20 s), `node eval_policy.mjs <policy.json> 8 "0.6,1"` (deterministic evaluation; `straight` gives the
zero-action baseline) and `node dashboard.mjs` (scalars & losses of every run at http://127.0.0.1:8789/). Copy the
best policy into `model/policy.json` (belt) or `model/policy_atmo.json` (atmosphere).
