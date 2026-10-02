# D104 — Snapshot/restore reproducible de MLflow con DVC

## Flujo

## Nota sobre el escaneo de secretos

Un `grep -i "secret"` sobre `db/mlflow.sql` encuentra coincidencias — son el
**esquema** de la tabla nativa `secrets` de MLflow 3.x (usada para
credenciales de proveedores GenAI externos, no usada en este proyecto:
solo tracking de entrenamiento). Confirmado vacía por dos vías
independientes antes de publicar:

    grep -c "INSERT INTO \`secrets\`" data/mlflow_snapshot/db/mlflow.sql   # 0
    docker compose exec mariadb mariadb -uroot -p"$MARIADB_ROOT_PASSWORD" \
      -e "SELECT COUNT(*) FROM mlflow.secrets;"                            # 0

Si en el futuro esta tabla llegara a tener filas, excluir sus DATOS del
dump (sin tocar el resto) con:

    mariadb-dump ... --ignore-table=mlflow.secrets --databases mlflow > ...
    mariadb-dump ... --no-data mlflow secrets >> ...   # solo el esquema, sin filas