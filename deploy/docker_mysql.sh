docker run \
--name mysql \
-d \
-p 3306:3306 \
-e MYSQL_ROOT_PASSWORD=123456 \
mysql