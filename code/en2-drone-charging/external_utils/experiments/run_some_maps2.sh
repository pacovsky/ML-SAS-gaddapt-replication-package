#!/bin/bash
source /root/EEEE/VENV_DIR/bin/activate
cmd_file="/root/redflags-honza/external_utils/experiments/new_commands_charged_ens_in_composition.run"
cmd_file='/root/redflags-honza/external_utils/experiments/new_commands_charged_ens_in_composition_final_gen_anim.run'
cmd_file='/root/redflags-honza/external_utils/experiments/new_commands_charged_ens_in_composition_final_gen.run'
#cmd_file='/root/redflags-honza/external_utils/experiments/new_commands_charged_ens_in_composition_final_gen_anim_another2.run' #another charging ensemble
cmd_file="/root/redflags-honza/external_utils/experiments/final_commands.run"
cmd_file='/root/redflags-honza/external_utils/experiments/new_commands_charged_ens_in_composition_final_gen_obey.run'
echo $cmd_file
cat $cmd_file | xargs --max-procs=78 -I CMD bash -c "python CMD " > /dev/null;
date +"%Y-%m-%d %T"
