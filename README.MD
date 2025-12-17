Note to self:
mlflow experiments create --experiment-name ${{ env.EXPERIMENT_NAME }} || true
mlflow run MLProject --experiment-name ${{ env.EXPERIMENT_NAME }} --env-manager=local

run the command on top on actions so that the ID will be detected on the same experiment