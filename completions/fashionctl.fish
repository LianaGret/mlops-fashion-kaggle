# Disable file completions by default
complete -c fashionctl -f

# Main commands
complete -c fashionctl -n "__fish_use_subcommand" -a "datasets" -d "Manage datasets"
complete -c fashionctl -n "__fish_use_subcommand" -a "models" -d "Manage models"
complete -c fashionctl -n "__fish_use_subcommand" -a "version" -d "Show version"

# datasets subcommands
complete -c fashionctl -n "__fish_seen_subcommand_from datasets" -a "download" -d "Download datasets"
complete -c fashionctl -n "__fish_seen_subcommand_from datasets" -a "list" -d "List available datasets"
complete -c fashionctl -n "__fish_seen_subcommand_from datasets" -a "info" -d "Show dataset info"

# datasets download options
complete -c fashionctl -n "__fish_seen_subcommand_from download" -l name -d "Dataset name" -xa "articles customers transactions images"
complete -c fashionctl -n "__fish_seen_subcommand_from download" -l all -d "Download all datasets"
complete -c fashionctl -n "__fish_seen_subcommand_from download" -l force -d "Force re-download"

# datasets info options
complete -c fashionctl -n "__fish_seen_subcommand_from info" -xa "articles customers transactions images" -d "Dataset name"

# models subcommands
complete -c fashionctl -n "__fish_seen_subcommand_from models" -a "train" -d "Train a model"
complete -c fashionctl -n "__fish_seen_subcommand_from models" -a "infer" -d "Run inference"
complete -c fashionctl -n "__fish_seen_subcommand_from models" -a "list" -d "List checkpoints"
complete -c fashionctl -n "__fish_seen_subcommand_from models" -a "export" -d "Export to ONNX"

# models train options
complete -c fashionctl -n "__fish_seen_subcommand_from train" -l config -d "Config name" -xa "train debug"
complete -c fashionctl -n "__fish_seen_subcommand_from train" -l overrides -d "Config overrides"
complete -c fashionctl -n "__fish_seen_subcommand_from train" -l resume -d "Resume from checkpoint" -F
complete -c fashionctl -n "__fish_seen_subcommand_from train" -l dry-run -d "Print config only"

# models infer options
complete -c fashionctl -n "__fish_seen_subcommand_from infer" -l checkpoint -d "Checkpoint path" -F
complete -c fashionctl -n "__fish_seen_subcommand_from infer" -l model-type -d "Model type" -xa "baseline collaborative content hybrid"
complete -c fashionctl -n "__fish_seen_subcommand_from infer" -l input-path -d "Input data path" -F
complete -c fashionctl -n "__fish_seen_subcommand_from infer" -l output-path -d "Output path" -F
complete -c fashionctl -n "__fish_seen_subcommand_from infer" -l batch-size -d "Batch size"

# models export options
complete -c fashionctl -n "__fish_seen_subcommand_from export" -l checkpoint -d "Checkpoint path" -F
complete -c fashionctl -n "__fish_seen_subcommand_from export" -l output -d "Output ONNX path" -F
complete -c fashionctl -n "__fish_seen_subcommand_from export" -l opset-version -d "ONNX opset version"
