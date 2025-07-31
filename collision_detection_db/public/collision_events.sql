create table collision_events
(
    id            serial
        primary key,
    timestamp     timestamp default CURRENT_TIMESTAMP,
    event_type    varchar(20) not null,
    probability   double precision,
    distance      double precision,
    object1_id    integer,
    object2_id    integer,
    object1_class varchar(20),
    object2_class varchar(20),
    frame_number  integer,
    severity      varchar(10),
    resolved      boolean   default false,
    created_at    timestamp default CURRENT_TIMESTAMP,
    updated_at    timestamp default CURRENT_TIMESTAMP
);

alter table collision_events
    owner to collision_admin;

create index idx_collision_events_event_type
    on collision_events (event_type);

create index idx_collision_events_timestamp
    on collision_events (timestamp);

