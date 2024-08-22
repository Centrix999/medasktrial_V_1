PostgreSQL database:
super admin user name: default:postgress
pass word:PostgreSQL password
-- data base name:centrix_db_001;
-- user:centrix,
-- schema:centrix_db_001_schema


-- creation of database:
CREATE DATABASE centrix_db_001;
-- moving to selected db:
postgres=# \c centrix_db_001

-- creation of pass word for user:
CREATE USER centrix_001 WITH PASSWORD 'centrix_001@1';


-- You are now connected to database "centrix_db" as user "postgres".

-- creation of schema: name:centrix_db_schema ::
centrix_db_001=# CREATE SCHEMA centrix_db_001_schema AUTHORIZATION centrix_001;




-- modify the user role settings:
-- 1.client_encodings  ='utf8'
-- 2.default_transaction_isolation= 'read committed'
-- 3.timezone= 'UTC'
-- 4. permissions:search_path = centrix_db_schema;


centrix_db_001=# ALTER ROLE centrix_001 SET client_encoding TO 'utf8';
ALTER ROLE

centrix_db_001=# ALTER ROLE centrix_001 SET default_transaction_isolation TO 'read committed';
ALTER ROLE

centrix_db_001=# ALTER ROLE centrix_001 SET timezone TO 'UTC'; 

centrix_db_001=# ALTER ROLE centrix_001 IN DATABASE centrix_db_001_db set search_path = centrix_db_001_schema;
ALTER ROLE


then do migrate next makemigrations

\l list of databases
\q:quit
\dt:data base table list
\du data base users
\dn:list of schemas 


note:
grant permissions:
GRANT CONNECT ON DATABASE centrix_db_001 TO centrix_001;

GRANT USAGE ON SCHEMA centrix_db_001_schema TO centrix_001;

GRANT ALL PRIVILEGES ON DATABASE centrix_db_001 TO centrix_001;


------------------

ALTER SCHEMA public OWNER TO centrix_001;
GRANT ALL ON SCHEMA public TO centrix_001;


