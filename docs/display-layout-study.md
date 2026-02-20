# Display and layout study — fit all tabs at 1080p and scale up

**Goal:** All tabs display in their entirety in fullscreen windowed at 1080p (1920×1080), scale sensibly to higher resolutions, and minimize horizontal/vertical scrolling.

This document is the reference for implementing and maintaining the GUI layout. Its conclusions are integrated into the gameplan as the **general rule** for any new tab or element.

---

## 1. Current state — what affects display and scroll

### 1.1 Main window (`main.py`)

| Element | Current behavior | Effect on scroll |
|--------|-------------------|------------------|
| Minimum size | `setMinimumSize(1920, 1080)` | Prevents window smaller than 1080p. |
| Initial size | `resize(1920, 1080)` | Starts at 1080p. |
| Show | `showMaximized()` | On 1080p screen, client area is **less than** 1920×1080 (title bar, tab bar, taskbar). |
| Central widget | `QTabWidget` (no explicit size policy) | Tab content gets whatever the main window client area is. |

**Approximate client area at 1080p maximized (Windows):** ~1920×~1005 (title bar ~32px, tab bar ~35px, optional taskbar). So **vertical space for tab content is ~1005px**, not 1080px.

### 1.2 League Parameters tab (`league_tab.py`)

| Element | Current behavior | Effect on scroll |
|--------|-------------------|------------------|
| Scroll area | `QScrollArea`, `setWidgetResizable(True)`, scroll bars as needed | Any content larger than viewport → scroll. |
| Content widget | `setMinimumWidth(1920)`, `setMinimumHeight(1000)` when project loaded | **Forces horizontal scroll** if viewport width &lt; 1920; **vertical scroll** if viewport height &lt; 1000. |
| No-project state | Content width locked to viewport | No horizontal scroll when no project. |
| Fixed heights | Project 84px, Population 400px, Arrow 14px, Flow boxes 526px each | Total content height = 1000px + margins; tight fit. |
| Resize callback | Viewport resize → content min height = max(1000, viewport height) | Fills vertical space when window is tall; doesn’t remove horizontal scroll. |

**Root cause of League tab scroll:** Content is **fixed at 1920px width**. On a 1920-wide screen the viewport is often slightly narrower (borders, DPI); on smaller or split windows horizontal scroll is guaranteed.

### 1.3 Dashboard tab (`main.py`)

| Element | Current behavior | Effect on scroll |
|--------|-------------------|------------------|
| Layout | `QVBoxLayout`: label + RunSectionWidget + stretch | No scroll area; fills available height. |
| RunSectionWidget | Run group + charts placeholder (min height 180, stretch 1) | Can grow; no fixed content height. |
| No minimum content size | — | Doesn’t force scroll; if we add more content later, we’ll need a strategy. |

Dashboard doesn’t cause scroll today but has no “fit to 1080p” design.

### 1.4 Agents, Play, Settings tabs

Placeholder layouts (label + widgets + stretch). No scroll areas, no fixed sizes. Future content could overflow without a shared rule.

### 1.5 Chart widgets (`charts.py`)

- **PopulationPieWidget:** min 220×160, max 320×200.
- **FitnessVisualWidget:** min 520×80 (and similar).
- **MutationDistWidget:** min 260×200, max height 280.
- **ReproductionBarWidget:** min height 56.

These live inside the League tab’s fixed-height flow boxes; they don’t directly cause window-level scroll but must stay within their box.

### 1.6 Dialogs

- **GroupDetailDialog:** min 900×400 (own window; scroll is inside dialog if needed).
- **NewProjectDialog:** min 450×380.

Not relevant to “tabs in their entirety” but good to keep reasonable.

---

## 2. Why scrolling happens (summary)

1. **League tab, project loaded:** Content width is **1920px** → horizontal scroll whenever viewport width &lt; 1920 (e.g. window not full screen, or DPI scaling reducing logical width).
2. **League tab, vertical:** Content height **1000px** with viewport ~1005px → very tight; any extra margin or tab bar variation can cause a few pixels of vertical scroll.
3. **Other tabs:** No explicit “fit in viewport” rule; future additions may overflow.
4. **Single design target:** Layout was designed for “1080p” but the **actual available height** (client area) is smaller than 1080, and width can be smaller than 1920 in real use.

---

## 3. Options and ways to implement

### Option A — Fit to viewport (responsive, minimize scroll)

**Idea:** Tab content never forces a larger size than the viewport. At 1080p fullscreen, everything fits without scroll; on larger screens content uses the space (or stays fixed and centered). No fixed 1920px width.

**Ways to implement:**

1. **Main window / tab area**
   - Keep `setMinimumSize(1920, 1080)` and `showMaximized()`.
   - Ensure the tab widget’s **current page** gets a resize when the window resizes (default with `setCentralWidget(tabs)`).
2. **League tab**
   - Remove `content.setMinimumWidth(1920)` when project loaded. Use either:
     - **A1:** Content width = viewport width (same as no-project): `content.setMinimumWidth(vw)`, `content.setMaximumWidth(vw)` always, and make the **layout** flexible (flow boxes and table use horizontal space proportionally). No horizontal scroll.
     - **A2:** Content min width = `min(1920, viewport width)` so on small viewports content shrinks and reflows (e.g. flow boxes stack or shrink; table gets horizontal scroll only inside a dedicated scroll area for the table).
3. **Content height**
   - Set content min height to **viewport height** (already done for “fill when tall”). For “fit at 1080p”, ensure the **design** fits in ~1000px (e.g. reduce League fixed heights slightly so 84+400+14+4×box &lt; 1000, or make one section slightly flexible).
4. **Dashboard / Agents / Play / Settings**
   - Put each tab’s main content in a `QScrollArea` with `setWidgetResizable(True)`. Set the **inner** widget’s maximum width to the viewport width (on resize) so long text/lists don’t force window-wide horizontal scroll. Prefer vertical scroll only when content is tall.
5. **Single rule for new tabs**
   - “Tab content lives in a scroll area; inner content max width = viewport width; design for ~1000px height at 1080p so vertical scroll is rare.”

**Pros:** No horizontal scroll at 1080p; works on smaller or split windows.  
**Cons:** League layout must reflow or shrink (narrower flow boxes or stacked); some re-layout work.

---

### Option B — Keep fixed 1080p layout, reduce scroll triggers

**Idea:** Keep the current “1920×1000” design intent but tweak so that in **maximized 1080p** we stay just under the visible area (no scroll).

**Ways to implement:**

1. **Use “available” size instead of 1920**
   - On first show and on resize: get the **tab widget’s current page** size (or the scroll viewport size). Set content width to `min(1920, viewport width)` and content height to `min(1000, viewport height)`. So when viewport is 1920×1005 we use 1920×1000 and no scroll; when viewport is 1600×900 we use 1600×900 and no scroll (content would need to be responsive or accept clipping — see below).
2. **Slightly reduce fixed heights**
   - e.g. CONTENT_HEIGHT_1080P = 990, PROJECT_HEIGHT = 80, POPULATION_HEIGHT = 380, FLOW_BOX_HEIGHT computed so total &lt; 990. Leaves a few pixels margin under 1080p client area.
3. **Horizontal**
   - If we keep content at 1920 when project loaded, horizontal scroll on true 1080p fullscreen can still appear (e.g. window border). To avoid it: **cap content width at viewport width** when viewport width &lt; 1920 (content then 1600 or 1920 depending on window). That implies layout must look acceptable at 1600 (e.g. flow boxes narrower).
4. **Scroll bar policy**
   - Keep `ScrollBarAsNeeded`; optionally hide scroll bars when content size &lt;= viewport (Qt default) so when we’re exactly at the limit we don’t show a tiny scroll.

**Pros:** Minimal code change; preserves current “fixed” look when window is 1920+ wide.  
**Cons:** On narrow windows content is either clipped or we still scroll; need to define behavior when viewport &lt; 1920.

---

### Option C — Hybrid (recommended baseline)

**Idea:** Design to **fit in viewport** at 1080p and above; one clear rule for all tabs; League tab adapts width so horizontal scroll is never required at fullscreen 1080p.

**Concrete implementation plan:**

1. **Reference size**
   - Define **TARGET_VIEWPORT_1080P = (1920, 1000)** (width, height) as the design reference. All “fixed” section heights are chosen so total content height ≤ 1000.
2. **Main window**
   - Keep minimum size 1920×1080, maximized. No change.
3. **League tab**
   - **Width:** Always set content width to **viewport width** (not 1920). So: `content.setMinimumWidth(vw)`, `content.setMaximumWidth(vw)` for both project-loaded and no-project. Layout uses **flexible horizontal space** (e.g. flow row with stretch factors, table and pie use remaining space). Result: **no horizontal scroll**.
   - **Height:** Content min height = `max(1000, viewport height)` so when window is tall we fill; when at 1080p we have 1000px content and ~1005 viewport → no vertical scroll.
   - **Section heights:** Keep fixed (Project, Population, Arrow, Flow boxes) so they sum to 1000; optionally reduce by 5–10px if any OS still shows a sliver of vertical scroll.
4. **Dashboard**
   - Wrap content in `QScrollArea`; inner widget max width = viewport width. Reserve a fixed height for run section + charts (e.g. run 120px, charts 400px) so at 1080p the tab fits without scroll; if content grows later, only vertical scroll.
5. **Agents, Play, Settings**
   - Same pattern: scroll area, inner max width = viewport, design for ~1000px height. New widgets follow this rule.
6. **Rule for new tabs/elements (for GAMEPLAN)**
   - See §5 below.

**Pros:** One rule; no horizontal scroll; predictable at 1080p and scalable.  
**Cons:** League flow boxes become narrower when viewport &lt; 1920 (acceptable if we use proportional layout).

---

### Option D — Scale factor for higher resolutions

**Idea:** At 1080p we fit as in Option C; on WQHD (2560×1440) or 4K we **scale** the same layout (bigger fonts/widgets) or add more content. “Scalable to superior resolutions” can mean:

- **D1 — Same layout, larger window:** No code change; just more empty space or same content centered. Already the case.
- **D2 — Scale UI (DPI / font scaling):** Rely on Qt/OS DPI awareness; set `Qt.AA_EnableHighDpiScaling` (or Qt 6 high-DPI) so at 125%/150% the layout scales. Test at 1080p and 1440p.
- **D3 — More content on big screens:** e.g. show more table rows or a second chart when height &gt; 1200. Optional; not required for “fit at 1080p and scale.”

Recommendation: Start with D1; ensure DPI scaling (D2) is enabled so that “scale to superior resolutions” is smooth without extra code.

---

## 4. Implementation checklist (Option C)

- [ ] **Constants:** In one place (e.g. `league_tab` or a small `layout_constants.py`), define `VIEWPORT_WIDTH_1080P = 1920`, `VIEWPORT_HEIGHT_1080P = 1000`, and section heights that sum to 1000.
- [ ] **League tab — width:** Remove the “when project loaded set min/max width 1920/unlimited”. Always set content width to viewport width (update in resize callback and in `_update_content_visibility`). Ensure the flow row and population area use stretch so they share width; table gets horizontal scroll only inside its own area if needed (optional: `QTableWidget` in a horizontal scroll area only for the table).
- [ ] **League tab — height:** Keep content min height = max(1000, viewport height). Optionally reduce total from 1000 to 995 to add margin.
- [ ] **Dashboard:** Add `QScrollArea`; set inner widget’s maximum width from viewport on resize; fix run + charts heights so total fits in 1000px.
- [ ] **Agents / Play / Settings:** When implementing real content, use scroll area + viewport-width cap + target height ≤ 1000px.
- [ ] **New elements:** Follow the rule in §5; avoid fixed widths &gt; viewport; prefer flexible layouts (stretch, size policies).

---

## 5. General rule for the gameplan (new tab or element)

**Display and layout rule (integrate into GAMEPLAN §8):**

1. **Target viewport:** 1080p fullscreen windowed → design for a **content area of 1920×1000** (width × height). Actual viewport may be ~1920×1005; keep total content height ≤ 1000 so vertical scroll is not required.
2. **No forced horizontal scroll:** Tab content must never set a minimum width greater than the **current viewport width**. Use viewport width as the content width (or max width) so the user never has to scroll sideways for the main layout.
3. **Structure per tab:** Each tab’s main content lives in a **QScrollArea** (or equivalent) with an inner widget. The inner widget’s **maximum width** is set to the viewport width on resize so that only **vertical** scroll appears when content is tall.
4. **Fixed heights:** Use fixed heights for key sections (e.g. Project bar, Population block, flow boxes) so the layout is predictable and fits in 1000px. Use stretch only for one "fill" section if needed. Use stretch only for one “fill” section if needed.
5. **Higher resolutions:** Same layout; no fixed pixel width &gt; viewport. Rely on Qt/OS DPI scaling for larger screens; optional “more content when tall” later.
6. **New widgets:** Prefer minimum sizes and expanding size policies; avoid `setMinimumWidth(1920)` or any width larger than the parent/viewport. Use `addStretch()` and stretch factors so content reflows within the available width.

---

## 6. Files to touch (Option C)

| File | Change |
|------|--------|
| `src/tarot_gui/league_tab.py` | Content width always = viewport width; remove 1920 min when project loaded; ensure flow/population layout flexible. |
| `src/tarot_gui/main.py` | Dashboard (and later Agents/Play/Settings): wrap in scroll area, set inner max width from viewport. |
| `GAMEPLAN.md` | Add “Display and layout rule” under §8 (Phase 6 — GUI), referencing this doc. |
| Optional: `src/tarot_gui/layout_constants.py` | Centralize VIEWPORT_HEIGHT_1080P, section heights, for reuse. |

---

*Document version: 1.0 — Options A–D, recommended Option C, and rule for gameplan integration.*
