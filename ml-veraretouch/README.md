# VeraRetouch

Standalone VeraRetouch app for on-device Apple Silicon inference.

<table>
<tr>
    <td><img src="../docs/veraretouch-counting.gif" alt="VeraRetouch - Counting"></td>
    <td><img src="../docs/veraretouch-handwriting.gif" alt="VeraRetouch - Handwriting"></td>
    <td><img src="../docs/veraretouch-emoji.gif" alt="VeraRetouch - Emoji"></td>
</tr>
</table>

## Features

- Runs independently from `app/`.
- Supports VeraRetouch checkpoints with `model_type` values `samantha`, `fastvlm`, or `llava_qwen2`.
- Uses the same on-device MLX + CoreML pipeline as the VeraRetouch reference app.

### Flexible Prompting

<img src="../docs/veraretouch-flexible_prompts.png" alt="Flexible prompting" style="width:66%;">

The app includes a set of built-in prompts to help you get started quickly. Tap the **Prompts** button in the top-right corner to explore them. Selecting a prompt will immediately update the active input. To create new prompts or edit existing ones, choose **Customize…** from the **Prompts** menu.

## Model Setup

1. Put your Samantha converted model files into:
`app_samantha_mac/VeraRetouchCore/model`
2. Ensure `config.json` and tokenizer files are present, plus `fastvithd.mlpackage`.
3. Open `app_samantha_mac/VeraRetouch.xcodeproj` in Xcode, build, and run.
