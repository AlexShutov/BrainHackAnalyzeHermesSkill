# Comment gate: what to check before reporting "no comments"

An empty comment block on a video that obviously has comments is almost never absence — it is a gate. Check in this order and stop at the first step that fires; each step answers a different question.

1. **Comments disabled on the video** — the fetch script says so explicitly. Report that, don't fabricate percentages.
2. **Anonymous session.** The innertube comments continuation returns key-based `commentViewModel` + `preloadMessageNames` but NO `commentEntityPayload` and no usable `frameworkUpdates` — keys without text. In the DOM this is indistinguishable from "no comments": the section header renders and the thread list stays empty even in a real browser on a video with millions of comments. So an empty block on a big video = auth gate, not absence.
3. **Login test by cookie names** (cheap, decisive): `SID`/`HSID`/`SSID`/`LOGIN_INFO` present → authenticated. Only `VISITOR_INFO1_LIVE`/`YSC`/`PREF`/`GPS` → anonymous. Login is the user's action; never type credentials, check the vault first.
4. **JS challenge (separate blocker).** Once cookies are present, an innertube call without attestation returns HTTP 400 `failedPrecondition`. yt-dlp reports `n challenge solving failed` unless a JS runtime solves the challenge: deno is the default; with node the solver must be downloaded (`--remote-components ejs:github`). The info/video path can solve the challenge cleanly while the comments continuation still returns nothing — that combination means the session is anonymous, not that the solver is broken.
5. **A long hang on the comments continuation is the loop guard firing** (`Detected YouTube comments looping`), not a network timeout. Report it as the gate, never as "the network is slow" and never as a script bug.

## Working path once the user has logged in

Dump the session cookies through CDP (Chrome encrypts its cookie DB; let Chrome decrypt them itself), convert to Netscape format, and pass them to yt-dlp:

- `Network.getCookies` with `urls=['https://www.youtube.com']` → write `# Netscape HTTP Cookie File` then one line per cookie: `domain\tTRUE\tpath\tTRUE\t<1 if secure else 0>\t<expiry unix>\tname\tvalue`; HttpOnly cookies get the `#HttpOnly_` prefix on the domain field.
- `PYTHONPATH=<isolated yt-pylib> python -m yt_dlp --skip-download --write-comments --js-runtimes node --remote-components ejs:github --cookies <file> -j URL`
- Without a login the only remaining route is the official Data API `commentThreads` (needs an API key — never guess one).

## Test-side consequence

A network script is only testable when it has an offline/dry-run mode and takes its client from outside (mock-transport). The dry-run must run BEFORE any session/login check — otherwise the offline mode, which is the only mode a test can use, returns `login_required` on a machine with no session. For network-dependent scripts the read-only live check is the dry-run, not the network call.
