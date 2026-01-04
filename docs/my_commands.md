-- Source - https://stackoverflow.com/a
-- Posted by Jim Jones, modified by community. See post 'Timeline' for change history
-- Retrieved 2026-01-02, License - CC BY-SA 4.0

docker exec -it -u root fe6d14f5f75c \
psql -d oi_db_new -c "COPY (SELECT * FROM nse_multi_expiry_minute_data) TO STDOUT CSV" > nse_multi_expiry_minute_data.csv
