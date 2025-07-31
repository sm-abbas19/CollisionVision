create table system_metrics
(
    id                     serial
        primary key,
    timestamp              timestamp default CURRENT_TIMESTAMP,
    fps                    double precision,
    total_objects_detected integer,
    active_tracks          integer,
    memory_usage           double precision,
    cpu_usage              double precision,
    created_at             timestamp default CURRENT_TIMESTAMP
);

alter table system_metrics
    owner to collision_admin;

create index idx_system_metrics_timestamp
    on system_metrics (timestamp);

