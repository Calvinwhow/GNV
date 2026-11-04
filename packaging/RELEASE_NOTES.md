## Download

| Your computer | File |
|---|---|
| Mac with Apple Silicon (M1 or later) | `GeneralNeuroimageViewer-macOS-arm64.dmg` |
| Mac with an Intel processor | `GeneralNeuroimageViewer-macOS-x64.dmg` |
| Windows | `GeneralNeuroimageViewer-Windows-x64.zip` |

Not sure which Mac you have? Apple menu → About This Mac: "Chip: Apple M…" is Apple Silicon, "Processor: … Intel" is Intel.

## Install

**Mac**

1. Open the `.dmg` and drag **General Neuroimage Viewer** onto **Applications**.
2. Open it from Applications. The first time, macOS says it "could not verify" the app is free of malware, because the app is not yet signed with an Apple developer certificate. Click **Done**.
3. Open **System Settings → Privacy & Security**, scroll down to the line about General Neuroimage Viewer, and click **Open Anyway**. Then open the app again and choose **Open**. You only do this once.

Or, instead of step 3, run this once in Terminal:

```bash
xattr -dr com.apple.quarantine "/Applications/General Neuroimage Viewer.app"
```

**Windows**

1. Unzip `GeneralNeuroimageViewer-Windows-x64.zip` anywhere and run `GeneralNeuroimageViewer.exe` inside it.
2. If Windows SmartScreen warns about an unrecognised app, click **More info → Run anyway**.

The viewer opens in a Chrome or Edge window of its own (or a tab in your default browser if you have neither).
