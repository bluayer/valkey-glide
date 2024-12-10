#![allow(unknown_lints, dependency_on_unit_never_type_fallback)]
#![cfg(feature = "cluster-async")]
mod support;

use std::cell::Cell;
use tokio::sync::Mutex;

use lazy_static::lazy_static;

lazy_static! {
    static ref CLUSTER_VERSION: Mutex<Cell<usize>> = Mutex::<Cell<usize>>::default();
}

/// Check if the current cluster version is less than `min_version`.
/// At first, the func check for the Valkey version and if none exists, then the Redis version is checked.
async fn engine_version_less_than(min_version: &str) -> bool {
    let test_version = crate::get_cluster_version().await;
    let min_version_usize = crate::version_to_usize(min_version).unwrap();
    if test_version < min_version_usize {
        println!(
            "The engine version is {:?}, which is lower than {:?}",
            test_version, min_version
        );
        return true;
    }
    false
}

/// Static function to get the engine version. When version looks like 8.0.0 -> 80000 and 12.0.1 -> 120001.
async fn get_cluster_version() -> usize {
    let cluster_version = CLUSTER_VERSION.lock().await;
    if cluster_version.get() == 0 {
        let cluster = crate::support::TestClusterContext::new(3, 0);

        let mut connection = cluster.async_connection(None).await;

        let cmd = redis::cmd("INFO");
        let info = connection
            .route_command(
                &cmd,
                redis::cluster_routing::RoutingInfo::SingleNode(
                    redis::cluster_routing::SingleNodeRoutingInfo::Random,
                ),
            )
            .await
            .unwrap();

        let info_result = redis::from_owned_redis_value::<String>(info).unwrap();

        cluster_version.set(
            parse_version_from_info(info_result.clone())
                .unwrap_or_else(|| panic!("Invalid version string in INFO : {info_result}")),
        );
    }
    cluster_version.get()
}

fn parse_version_from_info(info: String) -> Option<usize> {
    // check for valkey_version
    if let Some(version) = info
        .lines()
        .find_map(|line| line.strip_prefix("valkey_version:"))
    {
        return version_to_usize(version);
    }

    // check for redis_version if no valkey_version was found
    if let Some(version) = info
        .lines()
        .find_map(|line| line.strip_prefix("redis_version:"))
    {
        return version_to_usize(version);
    }
    None
}

/// Takes a version string (e.g., 8.2.1) and converts it to a usize (e.g., 80201)
/// version 12.10.0 will became 121000
fn version_to_usize(version: &str) -> Option<usize> {
    version
        .split('.')
        .enumerate()
        .map(|(index, part)| {
            part.parse::<usize>()
                .ok()
                .map(|num| num * 10_usize.pow(2 * (2 - index) as u32))
        })
        .sum()
}

#[cfg(test)]
mod cluster_async {
    use std::{
        collections::{HashMap, HashSet},
        net::{IpAddr, SocketAddr},
        str::from_utf8,
        sync::{
            atomic::{self, AtomicBool, AtomicI32, AtomicU16, AtomicU32, Ordering},
            Arc,
        },
        time::Duration,
    };

    use futures::prelude::*;
    use futures_time::{future::FutureExt, task::sleep};
    use once_cell::sync::Lazy;
    use std::ops::Add;

    use redis::{
        aio::{ConnectionLike, MultiplexedConnection},
        cluster::ClusterClient,
        cluster_async::{testing::MANAGEMENT_CONN_NAME, ClusterConnection, Connect},
        cluster_routing::{
            MultipleNodeRoutingInfo, Route, RoutingInfo, SingleNodeRoutingInfo, SlotAddr,
        },
        cluster_topology::{get_slot, DEFAULT_NUMBER_OF_REFRESH_SLOTS_RETRIES},
        cmd, from_owned_redis_value, parse_redis_value, AsyncCommands, Cmd, ErrorKind,
        FromRedisValue, GlideConnectionOptions, InfoDict, IntoConnectionInfo, ProtocolVersion,
        PubSubChannelOrPattern, PubSubSubscriptionInfo, PubSubSubscriptionKind, PushInfo, PushKind,
        RedisError, RedisFuture, RedisResult, Script, Value,
    };

    use crate::support::*;
    use tokio::sync::mpsc;
    fn broken_pipe_error() -> RedisError {
        RedisError::from(std::io::Error::new(
            std::io::ErrorKind::BrokenPipe,
            "mock-io-error",
        ))
    }

    fn validate_subscriptions(
        pubsub_subs: &PubSubSubscriptionInfo,
        notifications_rx: &mut mpsc::UnboundedReceiver<PushInfo>,
        allow_disconnects: bool,
    ) {
        let mut subscribe_cnt =
            if let Some(exact_subs) = pubsub_subs.get(&PubSubSubscriptionKind::Exact) {
                exact_subs.len()
            } else {
                0
            };

        let mut psubscribe_cnt =
            if let Some(pattern_subs) = pubsub_subs.get(&PubSubSubscriptionKind::Pattern) {
                pattern_subs.len()
            } else {
                0
            };

        let mut ssubscribe_cnt =
            if let Some(sharded_subs) = pubsub_subs.get(&PubSubSubscriptionKind::Sharded) {
                sharded_subs.len()
            } else {
                0
            };

        for _ in 0..(subscribe_cnt + psubscribe_cnt + ssubscribe_cnt) {
            let result = notifications_rx.try_recv();
            assert!(result.is_ok());
            let PushInfo { kind, data: _ } = result.unwrap();
            assert!(
                kind == PushKind::Subscribe
                    || kind == PushKind::PSubscribe
                    || kind == PushKind::SSubscribe
                    || if allow_disconnects {
                        kind == PushKind::Disconnection
                    } else {
                        false
                    }
            );
            if kind == PushKind::Subscribe {
                subscribe_cnt -= 1;
            } else if kind == PushKind::PSubscribe {
                psubscribe_cnt -= 1;
            } else if kind == PushKind::SSubscribe {
                ssubscribe_cnt -= 1;
            }
        }

        assert!(subscribe_cnt == 0);
        assert!(psubscribe_cnt == 0);
        assert!(ssubscribe_cnt == 0);
    }

    #[test]
    #[serial_test::serial]
    fn test_async_cluster_basic_cmd() {
        let cluster = TestClusterContext::new(3, 0);

        block_on_all(async move {
            let mut connection = cluster.async_connection(None).await;
            cmd("SET")
                .arg("test")
                .arg("test_data")
                .query_async(&mut connection)
                .await?;
            let res: String = cmd("GET")
                .arg("test")
                .clone()
                .query_async(&mut connection)
                .await?;
            assert_eq!(res, "test_data");
            Ok::<_, RedisError>(())
        })
        .unwrap();
    }

    #[tokio::test]
    async fn test_routing_by_slot_to_replica_with_az_affinity_strategy_to_half_replicas() {
        // Skip test if version is less then Valkey 8.0
        if crate::engine_version_less_than("8.0").await {
            return;
        }

        let replica_num: u16 = 4;
        let primaries_num: u16 = 3;
        let replicas_num_in_client_az = replica_num / 2;
        let cluster =
            TestClusterContext::new((replica_num * primaries_num) + primaries_num, replica_num);
        let az: String = "us-east-1a".to_string();

        let mut connection = cluster.async_connection(None).await;
        let cluster_addresses: Vec<_> = cluster
            .cluster
            .servers
            .iter()
            .map(|server| server.connection_info())
            .collect();

        let mut cmd = redis::cmd("CONFIG");
        cmd.arg(&["SET", "availability-zone", &az.clone()]);

        for _ in 0..replicas_num_in_client_az {
            connection
                .route_command(
                    &cmd,
                    RoutingInfo::SingleNode(SingleNodeRoutingInfo::SpecificNode(Route::new(
                        12182, // foo key is mapping to 12182 slot
                        SlotAddr::ReplicaRequired,
                    ))),
                )
                .await
                .unwrap();
        }

        let mut client = ClusterClient::builder(cluster_addresses.clone())
            .read_from(redis::cluster_slotmap::ReadFromReplicaStrategy::AZAffinity(
                az.clone(),
            ))
            .build()
            .unwrap()
            .get_async_connection(None)
            .await
            .unwrap();

        // Each replica in the client az will return the value of foo n times
        let n = 4;
        for _ in 0..n * replicas_num_in_client_az {
            let mut cmd = redis::cmd("GET");
            cmd.arg("foo");
            let _res: RedisResult<Value> = cmd.query_async(&mut client).await;
        }

        let mut cmd = redis::cmd("INFO");
        cmd.arg("ALL");
        let info = connection
            .route_command(
                &cmd,
                RoutingInfo::MultiNode((MultipleNodeRoutingInfo::AllNodes, None)),
            )
            .await
            .unwrap();

        let info_result = redis::from_owned_redis_value::<HashMap<String, String>>(info).unwrap();
        let get_cmdstat = "cmdstat_get:calls=".to_string();
        let n_get_cmdstat = format!("cmdstat_get:calls={}", n);
        let client_az = format!("availability_zone:{}", az);

        let mut matching_entries_count: usize = 0;

        for value in info_result.values() {
            if value.contains(&get_cmdstat) {
                if value.contains(&client_az) && value.contains(&n_get_cmdstat) {
                    matching_entries_count += 1;
                } else {
                    panic!(
                        "Invalid entry found: {}. Expected cmdstat_get:calls={} and availability_zone={}",
                        value, n, az);
                }
            }
        }

        assert_eq!(
            (matching_entries_count.try_into() as Result<u16, _>).unwrap(),
            replicas_num_in_client_az,
            "Test failed: expected exactly '{}' entries with '{}' and '{}', found {}",
            replicas_num_in_client_az,
            get_cmdstat,
            client_az,
            matching_entries_count
        );
    }

    #[tokio::test]
    async fn test_routing_by_slot_to_replica_with_az_affinity_strategy_to_all_replicas() {
        // Skip test if version is less then Valkey 8.0
        if crate::engine_version_less_than("8.0").await {
            return;
        }

        let replica_num: u16 = 4;
        let primaries_num: u16 = 3;
        let cluster =
            TestClusterContext::new((replica_num * primaries_num) + primaries_num, replica_num);
        let az: String = "us-east-1a".to_string();

        let mut connection = cluster.async_connection(None).await;
        let cluster_addresses: Vec<_> = cluster
            .cluster
            .servers
            .iter()
            .map(|server| server.connection_info())
            .collect();

        let mut cmd = redis::cmd("CONFIG");
        cmd.arg(&["SET", "availability-zone", &az.clone()]);

        connection
            .route_command(
                &cmd,
                RoutingInfo::MultiNode((MultipleNodeRoutingInfo::AllNodes, None)),
            )
            .await
            .unwrap();

        let mut client = ClusterClient::builder(cluster_addresses.clone())
            .read_from(redis::cluster_slotmap::ReadFromReplicaStrategy::AZAffinity(
                az.clone(),
            ))
            .build()
            .unwrap()
            .get_async_connection(None)
            .await
            .unwrap();

        // Each replica will return the value of foo n times
        let n = 4;
        for _ in 0..(n * replica_num) {
            let mut cmd = redis::cmd("GET");
            cmd.arg("foo");
            let _res: RedisResult<Value> = cmd.query_async(&mut client).await;
        }

        let mut cmd = redis::cmd("INFO");
        cmd.arg("ALL");
        let info = connection
            .route_command(
                &cmd,
                RoutingInfo::MultiNode((MultipleNodeRoutingInfo::AllNodes, None)),
            )
            .await
            .unwrap();

        let info_result = redis::from_owned_redis_value::<HashMap<String, String>>(info).unwrap();
        let get_cmdstat = "cmdstat_get:calls=".to_string();
        let n_get_cmdstat = format!("cmdstat_get:calls={}", n);
        let client_az = format!("availability_zone:{}", az);

        let mut matching_entries_count: usize = 0;

        for value in info_result.values() {
            if value.contains(&get_cmdstat) {
                if value.contains(&client_az) && value.contains(&n_get_cmdstat) {
                    matching_entries_count += 1;
                } else {
                    panic!(
                        "Invalid entry found: {}. Expected cmdstat_get:calls={} and availability_zone={}",
                        value, n, az);
                }
            }
        }

        assert_eq!(
            (matching_entries_count.try_into() as Result<u16, _>).unwrap(),
            replica_num,
            "Test failed: expected exactly '{}' entries with '{}' and '{}', found {}",
            replica_num,
            get_cmdstat,
            client_az,
            matching_entries_count
        );
    }

    #[test]
    #[serial_test::serial]
    fn test_async_cluster_basic_eval() {
        let cluster = TestClusterContext::new(3, 0);

        block_on_all(async move {
            let mut connection = cluster.async_connection(None).await;
            let res: String = cmd("EVAL")
                .arg(r#"redis.call("SET", KEYS[1], ARGV[1]); return redis.call("GET", KEYS[1])"#)
                .arg(1)
                .arg("key")
                .arg("test")
                .query_async(&mut connection)
                .await?;
            assert_eq!(res, "test");
            Ok::<_, RedisError>(())
        })
        .unwrap();
    }

    #[test]
    #[serial_test::serial]
    fn test_async_cluster_basic_script() {
        let cluster = TestClusterContext::new(3, 0);

        block_on_all(async move {
            let mut connection = cluster.async_connection(None).await;
            let res: String = Script::new(
                r#"redis.call("SET", KEYS[1], ARGV[1]); return redis.call("GET", KEYS[1])"#,
            )
            .key("key")
            .arg("test")
            .invoke_async(&mut connection)
            .await?;
            assert_eq!(res, "test");
            Ok::<_, RedisError>(())
        })
        .unwrap();
    }

    #[test]
    #[serial_test::serial]
    fn test_async_cluster_route_flush_to_specific_node() {
        let cluster = TestClusterContext::new(3, 0);

        block_on_all(async move {
            let mut connection = cluster.async_connection(None).await;
            let _: () = connection.set("foo", "bar").await.unwrap();
            let _: () = connection.set("bar", "foo").await.unwrap();

            let res: String = connection.get("foo").await.unwrap();
            assert_eq!(res, "bar".to_string());
            let res2: Option<String> = connection.get("bar").await.unwrap();
            assert_eq!(res2, Some("foo".to_string()));

            let route =
                redis::cluster_routing::Route::new(1, redis::cluster_routing::SlotAddr::Master);
            let single_node_route =
                redis::cluster_routing::SingleNodeRoutingInfo::SpecificNode(route);
            let routing = RoutingInfo::SingleNode(single_node_route);
            assert_eq!(
                connection
                    .route_command(&redis::cmd("FLUSHALL"), routing)
                    .await
                    .unwrap(),
                Value::Okay
            );
            let res: String = connection.get("foo").await.unwrap();
            assert_eq!(res, "bar".to_string());
            let res2: Option<String> = connection.get("bar").await.unwrap();
            assert_eq!(res2, None);
            Ok::<_, RedisError>(())
        })
        .unwrap();
    }

    #[test]
    #[serial_test::serial]
    fn test_async_cluster_route_flush_to_node_by_address() {
        let cluster = TestClusterContext::new(3, 0);

        block_on_all(async move {
            let mut connection = cluster.async_connection(None).await;
            let mut cmd = redis::cmd("INFO");
            // The other sections change with time.
            // TODO - after we remove support of redis 6, we can add more than a single section - .arg("Persistence").arg("Memory").arg("Replication")
            cmd.arg("Clients");
            let value = connection
                .route_command(
                    &cmd,
                    RoutingInfo::MultiNode((MultipleNodeRoutingInfo::AllNodes, None)),
                )
                .await
                .unwrap();

            let info_by_address = from_owned_redis_value::<HashMap<String, String>>(value).unwrap();
            // find the info of the first returned node
            let (address, info) = info_by_address.into_iter().next().unwrap();
            let mut split_address = address.split(':');
            let host = split_address.next().unwrap().to_string();
            let port = split_address.next().unwrap().parse().unwrap();

            let value = connection
                .route_command(
                    &cmd,
                    RoutingInfo::SingleNode(SingleNodeRoutingInfo::ByAddress { host, port }),
                )
                .await
                .unwrap();
            let new_info = from_owned_redis_value::<String>(value).unwrap();

            assert_eq!(new_info, info);
            Ok::<_, RedisError>(())
        })
        .unwrap();
    }

    #[test]
    #[serial_test::serial]
    fn test_async_cluster_route_info_to_nodes() {
        let cluster = TestClusterContext::new(12, 1);

        let split_to_addresses_and_info = |res| -> (Vec<String>, Vec<String>) {
            if let Value::Map(values) = res {
                let mut pairs: Vec<_> = values
                    .into_iter()
                    .map(|(key, value)| {
                        (
                            redis::from_redis_value::<String>(&key).unwrap(),
                            redis::from_redis_value::<String>(&value).unwrap(),
                        )
                    })
                    .collect();
                pairs.sort_by(|(address1, _), (address2, _)| address1.cmp(address2));
                pairs.into_iter().unzip()
            } else {
                unreachable!("{:?}", res);
            }
        };

        block_on_all(async move {
            let cluster_addresses: Vec<_> = cluster
                .cluster
                .servers
                .iter()
                .map(|server| server.connection_info())
                .collect();
            let client = ClusterClient::builder(cluster_addresses.clone())
                .read_from_replicas()
                .build()?;
            let mut connection = client.get_async_connection(None).await?;

            let route_to_all_nodes = redis::cluster_routing::MultipleNodeRoutingInfo::AllNodes;
            let routing = RoutingInfo::MultiNode((route_to_all_nodes, None));
            let res = connection
                .route_command(&redis::cmd("INFO"), routing)
                .await
                .unwrap();
            let (addresses, infos) = split_to_addresses_and_info(res);

            let mut cluster_addresses: Vec<_> = cluster_addresses
                .into_iter()
                .map(|info| info.addr.to_string())
                .collect();
            cluster_addresses.sort();

            assert_eq!(addresses.len(), 12);
            assert_eq!(addresses, cluster_addresses);
            assert_eq!(infos.len(), 12);
            for i in 0..12 {
                let split: Vec<_> = addresses[i].split(':').collect();
                assert!(infos[i].contains(&format!("tcp_port:{}", split[1])));
            }

            let route_to_all_primaries =
                redis::cluster_routing::MultipleNodeRoutingInfo::AllMasters;
            let routing = RoutingInfo::MultiNode((route_to_all_primaries, None));
            let res = connection
                .route_command(&redis::cmd("INFO"), routing)
                .await
                .unwrap();
            let (addresses, infos) = split_to_addresses_and_info(res);
            assert_eq!(addresses.len(), 6);
            assert_eq!(infos.len(), 6);
            // verify that all primaries have the correct port & host, and are marked as primaries.
            for i in 0..6 {
                assert!(cluster_addresses.contains(&addresses[i]));
                let split: Vec<_> = addresses[i].split(':').collect();
                assert!(infos[i].contains(&format!("tcp_port:{}", split[1])));
                assert!(infos[i].contains("role:primary") || infos[i].contains("role:master"));
            }

            Ok::<_, RedisError>(())
        })
        .unwrap();
    }

    #[test]
    #[serial_test::serial]
    fn test_async_cluster_resp3() {
        if use_protocol() == ProtocolVersion::RESP2 {
            return;
        }
        block_on_all(async move {
            let cluster = TestClusterContext::new(3, 0);

            let mut connection = cluster.async_connection(None).await;

            let hello: HashMap<String, Value> = redis::cmd("HELLO")
                .query_async(&mut connection)
                .await
                .unwrap();
            assert_eq!(hello.get("proto").unwrap(), &Value::Int(3));

            let _: () = connection.hset("hash", "foo", "baz").await.unwrap();
            let _: () = connection.hset("hash", "bar", "foobar").await.unwrap();
            let result: Value = connection.hgetall("hash").await.unwrap();

            assert_eq!(
                result,
                Value::Map(vec![
                    (
                        Value::BulkString("foo".as_bytes().to_vec()),
                        Value::BulkString("baz".as_bytes().to_vec())
                    ),
                    (
                        Value::BulkString("bar".as_bytes().to_vec()),
                        Value::BulkString("foobar".as_bytes().to_vec())
                    )
                ])
            );

            Ok(())
        })
        .unwrap();
    }

    #[test]
    #[serial_test::serial]
    fn test_async_cluster_basic_pipe() {
        let cluster = TestClusterContext::new(3, 0);

        block_on_all(async move {
            let mut connection = cluster.async_connection(None).await;
            let mut pipe = redis::pipe();
            pipe.add_command(cmd("SET").arg("test").arg("test_data").clone());
            pipe.add_command(cmd("SET").arg("{test}3").arg("test_data3").clone());
            pipe.query_async(&mut connection).await?;
            let res: String = connection.get("test").await?;
            assert_eq!(res, "test_data");
            let res: String = connection.get("{test}3").await?;
            assert_eq!(res, "test_data3");
            Ok::<_, RedisError>(())
        })
        .unwrap()
    }

    #[test]
    #[serial_test::serial]
    fn test_async_cluster_multi_shard_commands() {
        let cluster = TestClusterContext::new(3, 0);

        block_on_all(async move {
            let mut connection = cluster.async_connection(None).await;

            let res: String = connection
                .mset(&[("foo", "bar"), ("bar", "foo"), ("baz", "bazz")])
                .await?;
            assert_eq!(res, "OK");
            let res: Vec<String> = connection.mget(&["baz", "foo", "bar"]).await?;
            assert_eq!(res, vec!["bazz", "bar", "foo"]);
            Ok::<_, RedisError>(())
        })
        .unwrap()
    }

    #[test]
    #[serial_test::serial]
    fn test_async_cluster_basic_failover() {
        block_on_all(async move {
            test_failover(&TestClusterContext::new(6, 1), 10, 123, false).await;
            Ok::<_, RedisError>(())
        })
        .unwrap()
    }

    async fn do_failover(
        redis: &mut redis::aio::MultiplexedConnection,
    ) -> Result<(), anyhow::Error> {
        cmd("CLUSTER").arg("FAILOVER").query_async(redis).await?;
        Ok(())
    }

    // parameter `_mtls_enabled` can only be used if `feature = tls-rustls` is active
    #[allow(dead_code)]
    async fn test_failover(
        env: &TestClusterContext,
        requests: i32,
        value: i32,
        _mtls_enabled: bool,
    ) {
        let completed = Arc::new(AtomicI32::new(0));

        let connection = env.async_connection(None).await;
        let mut node_conns: Vec<MultiplexedConnection> = Vec::new();

        'outer: loop {
            node_conns.clear();
            let cleared_nodes = async {
                for server in env.cluster.iter_servers() {
                    let addr = server.client_addr();

                    #[cfg(feature = "tls-rustls")]
                    let client = build_single_client(
                        server.connection_info(),
                        &server.tls_paths,
                        _mtls_enabled,
                    )
                    .unwrap_or_else(|e| panic!("Failed to connect to '{addr}': {e}"));

                    #[cfg(not(feature = "tls-rustls"))]
                    let client = redis::Client::open(server.connection_info())
                        .unwrap_or_else(|e| panic!("Failed to connect to '{addr}': {e}"));

                    let mut conn = client
                        .get_multiplexed_async_connection(GlideConnectionOptions::default())
                        .await
                        .unwrap_or_else(|e| panic!("Failed to get connection: {e}"));

                    let info: InfoDict = redis::Cmd::new()
                        .arg("INFO")
                        .query_async(&mut conn)
                        .await
                        .expect("INFO");
                    let role: String = info.get("role").expect("cluster role");

                    if role == "master" {
                        tokio::time::timeout(std::time::Duration::from_secs(3), async {
                            Ok(redis::Cmd::new()
                                .arg("FLUSHALL")
                                .query_async(&mut conn)
                                .await?)
                        })
                        .await
                        .unwrap_or_else(|err| Err(anyhow::Error::from(err)))?;
                    }

                    node_conns.push(conn);
                }
                Ok::<_, anyhow::Error>(())
            }
            .await;
            match cleared_nodes {
                Ok(()) => break 'outer,
                Err(err) => {
                    // Failed to clear the databases, retry
                    tracing::warn!("{}", err);
                }
            }
        }

        (0..requests + 1)
            .map(|i| {
                let mut connection = connection.clone();
                let mut node_conns = node_conns.clone();
                let completed = completed.clone();
                async move {
                    if i == requests / 2 {
                        // Failover all the nodes, error only if all the failover requests error
                        let mut results = future::join_all(
                            node_conns
                                .iter_mut()
                                .map(|conn| Box::pin(do_failover(conn))),
                        )
                        .await;
                        if results.iter().all(|res| res.is_err()) {
                            results.pop().unwrap()
                        } else {
                            Ok::<_, anyhow::Error>(())
                        }
                    } else {
                        let key = format!("test-{value}-{i}");
                        cmd("SET")
                            .arg(&key)
                            .arg(i)
                            .clone()
                            .query_async(&mut connection)
                            .await?;
                        let res: i32 = cmd("GET")
                            .arg(key)
                            .clone()
                            .query_async(&mut connection)
                            .await?;
                        assert_eq!(res, i);
                        completed.fetch_add(1, Ordering::SeqCst);
                        Ok::<_, anyhow::Error>(())
                    }
                }
            })
            .collect::<stream::FuturesUnordered<_>>()
            .try_collect()
            .await
            .unwrap_or_else(|e| panic!("{e}"));

        assert_eq!(
            completed.load(Ordering::SeqCst),
            requests,
            "Some requests never completed!"
        );
    }

    static ERROR: Lazy<AtomicBool> = Lazy::new(Default::default);

    #[derive(Clone)]
    struct ErrorConnection {
        inner: MultiplexedConnection,
    }

    impl Connect for ErrorConnection {
        fn connect<'a, T>(
            info: T,
            response_timeout: std::time::Duration,
            connection_timeout: std::time::Duration,
            socket_addr: Option<SocketAddr>,
            glide_connection_options: GlideConnectionOptions,
        ) -> RedisFuture<'a, (Self, Option<IpAddr>)>
        where
            T: IntoConnectionInfo + Send + 'a,
        {
            Box::pin(async move {
                let (inner, _ip) = MultiplexedConnection::connect(
                    info,
                    response_timeout,
                    connection_timeout,
                    socket_addr,
                    glide_connection_options,
                )
                .await?;
                Ok((ErrorConnection { inner }, None))
            })
        }
    }

    impl ConnectionLike for ErrorConnection {
        fn req_packed_command<'a>(&'a mut self, cmd: &'a Cmd) -> RedisFuture<'a, Value> {
            if ERROR.load(Ordering::SeqCst) {
                Box::pin(async move { Err(RedisError::from((redis::ErrorKind::Moved, "ERROR"))) })
            } else {
                self.inner.req_packed_command(cmd)
            }
        }

        fn req_packed_commands<'a>(
            &'a mut self,
            pipeline: &'a redis::Pipeline,
            offset: usize,
            count: usize,
        ) -> RedisFuture<'a, Vec<Value>> {
            self.inner.req_packed_commands(pipeline, offset, count)
        }

        fn get_db(&self) -> i64 {
            self.inner.get_db()
        }

        fn is_closed(&self) -> bool {
            true
        }
    }

    #[test]
    #[serial_test::serial]
    fn test_async_cluster_error_in_inner_connection() {
        let cluster = TestClusterContext::new(3, 0);

        block_on_all(async move {
            let mut con = cluster.async_generic_connection::<ErrorConnection>().await;

            ERROR.store(false, Ordering::SeqCst);
            let r: Option<i32> = con.get("test").await?;
            assert_eq!(r, None::<i32>);

            ERROR.store(true, Ordering::SeqCst);

            let result: RedisResult<()> = con.get("test").await;
            assert_eq!(
                result,
                Err(RedisError::from((redis::ErrorKind::Moved, "ERROR")))
            );

            Ok::<_, RedisError>(())
        })
        .unwrap();
    }

    #[test]
    #[serial_test::serial]
    fn test_async_cluster_can_connect_to_server_that_sends_cluster_slots_without_host_name() {
        let name =
            "test_async_cluster_can_connect_to_server_that_sends_cluster_slots_without_host_name";

        let MockEnv {
            runtime,
            async_connection: mut connection,
            ..
        } = MockEnv::new(name, move |cmd: &[u8], _| {
            if contains_slice(cmd, b"PING") {
                Err(Ok(Value::SimpleString("OK".into())))
            } else if contains_slice(cmd, b"CLUSTER") && contains_slice(cmd, b"SLOTS") {
                Err(Ok(Value::Array(vec![Value::Array(vec![
                    Value::Int(0),
                    Value::Int(16383),
                    Value::Array(vec![
                        Value::BulkString("".as_bytes().to_vec()),
                        Value::Int(6379),
                    ]),
                ])])))
            } else {
                Err(Ok(Value::Nil))
            }
        });

        let value = runtime.block_on(
            cmd("GET")
                .arg("test")
                .query_async::<_, Value>(&mut connection),
        );

        assert_eq!(value, Ok(Value::Nil));
    }

    #[test]
    #[serial_test::serial]
    fn test_async_cluster_can_connect_to_server_that_sends_cluster_slots_with_null_host_name() {
        let name =
            "test_async_cluster_can_connect_to_server_that_sends_cluster_slots_with_null_host_name";

        let MockEnv {
            runtime,
            async_connection: mut connection,
            ..
        } = MockEnv::new(name, move |cmd: &[u8], _| {
            if contains_slice(cmd, b"PING") {
                Err(Ok(Value::SimpleString("OK".into())))
            } else if contains_slice(cmd, b"CLUSTER") && contains_slice(cmd, b"SLOTS") {
                Err(Ok(Value::Array(vec![Value::Array(vec![
                    Value::Int(0),
                    Value::Int(16383),
                    Value::Array(vec![Value::Nil, Value::Int(6379)]),
                ])])))
            } else {
                Err(Ok(Value::Nil))
            }
        });

        let value = runtime.block_on(
            cmd("GET")
                .arg("test")
                .query_async::<_, Value>(&mut connection),
        );

        assert_eq!(value, Ok(Value::Nil));
    }

    #[test]
    #[serial_test::serial]
    fn test_async_cluster_cannot_connect_to_server_with_unknown_host_name() {
        let name = "test_async_cluster_cannot_connect_to_server_with_unknown_host_name";
        let handler = move |cmd: &[u8], _| {
            if contains_slice(cmd, b"PING") {
                Err(Ok(Value::SimpleString("OK".into())))
            } else if contains_slice(cmd, b"CLUSTER") && contains_slice(cmd, b"SLOTS") {
                Err(Ok(Value::Array(vec![Value::Array(vec![
                    Value::Int(0),
                    Value::Int(16383),
                    Value::Array(vec![
                        Value::BulkString("?".as_bytes().to_vec()),
                        Value::Int(6379),
                    ]),
                ])])))
            } else {
                Err(Ok(Value::Nil))
            }
        };
        let client_builder = ClusterClient::builder(vec![&*format!("redis://{name}")]);
        let client: ClusterClient = client_builder.build().unwrap();
        let _handler = MockConnectionBehavior::register_new(name, Arc::new(handler));
        let connection = client.get_generic_connection::<MockConnection>(None);
        assert!(connection.is_err());
        let err = connection.err().unwrap();
        assert!(err
            .to_string()
            .contains("Error parsing slots: No healthy node found"))
    }

    #[test]
    #[serial_test::serial]
    fn test_async_cluster_can_connect_to_server_that_sends_cluster_slots_with_partial_nodes_with_unknown_host_name(
    ) {
        let name = "test_async_cluster_can_connect_to_server_that_sends_cluster_slots_with_partial_nodes_with_unknown_host_name";

        let MockEnv {
            runtime,
            async_connection: mut connection,
            ..
        } = MockEnv::new(name, move |cmd: &[u8], _| {
            if contains_slice(cmd, b"PING") {
                Err(Ok(Value::SimpleString("OK".into())))
            } else if contains_slice(cmd, b"CLUSTER") && contains_slice(cmd, b"SLOTS") {
                Err(Ok(Value::Array(vec![
                    Value::Array(vec![
                        Value::Int(0),
                        Value::Int(7000),
                        Value::Array(vec![
                            Value::BulkString(name.as_bytes().to_vec()),
                            Value::Int(6379),
                        ]),
                    ]),
                    Value::Array(vec![
                        Value::Int(7001),
                        Value::Int(16383),
                        Value::Array(vec![
                            Value::BulkString("?".as_bytes().to_vec()),
                            Value::Int(6380),
                        ]),
                    ]),
                ])))
            } else {
                Err(Ok(Value::Nil))
            }
        });

        let value = runtime.block_on(
            cmd("GET")
                .arg("test")
                .query_async::<_, Value>(&mut connection),
        );

        assert_eq!(value, Ok(Value::Nil));
    }

    #[test]
    #[serial_test::serial]
    fn test_async_cluster_retries() {
        let name = "tryagain";

        let requests = atomic::AtomicUsize::new(0);
        let MockEnv {
            runtime,
            async_connection: mut connection,
            handler: _handler,
            ..
        } = MockEnv::with_client_builder(
            ClusterClient::builder(vec![&*format!("redis://{name}")]).retries(5),
            name,
            move |cmd: &[u8], _| {
                respond_startup(name, cmd)?;

                match requests.fetch_add(1, atomic::Ordering::SeqCst) {
                    0..=4 => Err(parse_redis_value(b"-TRYAGAIN mock\r\n")),
                    _ => Err(Ok(Value::BulkString(b"123".to_vec()))),
                }
            },
        );

        let value = runtime.block_on(
            cmd("GET")
                .arg("test")
                .query_async::<_, Option<i32>>(&mut connection),
        );

        assert_eq!(value, Ok(Some(123)));
    }

    #[test]
    #[serial_test::serial]
    fn test_async_cluster_tryagain_exhaust_retries() {
        let name = "tryagain_exhaust_retries";

        let requests = Arc::new(atomic::AtomicUsize::new(0));

        let MockEnv {
            runtime,
            async_connection: mut connection,
            handler: _handler,
            ..
        } = MockEnv::with_client_builder(
            ClusterClient::builder(vec![&*format!("redis://{name}")]).retries(2),
            name,
            {
                let requests = requests.clone();
                move |cmd: &[u8], _| {
                    respond_startup(name, cmd)?;
                    requests.fetch_add(1, atomic::Ordering::SeqCst);
                    Err(parse_redis_value(b"-TRYAGAIN mock\r\n"))
                }
            },
        );

        let result = runtime.block_on(
            cmd("GET")
                .arg("test")
                .query_async::<_, Option<i32>>(&mut connection),
        );

        match result {
            Ok(_) => panic!("result should be an error"),
            Err(e) => match e.kind() {
                ErrorKind::TryAgain => {}
                _ => panic!("Expected TryAgain but got {:?}", e.kind()),
            },
        }
        assert_eq!(requests.load(atomic::Ordering::SeqCst), 3);
    }

    // Obtain the view index associated with the node with [called_port] port
    fn get_node_view_index(num_of_views: usize, ports: &Vec<u16>, called_port: u16) -> usize {
        let port_index = ports
            .iter()
            .position(|&p| p == called_port)
            .unwrap_or_else(|| {
                panic!(
                    "CLUSTER SLOTS was called with unknown port: {called_port}; Known ports: {:?}",
                    ports
                )
            });
        // If we have less views than nodes, use the last view
        if port_index < num_of_views {
            port_index
        } else {
            num_of_views - 1
        }
    }
    #[test]
    #[serial_test::serial]
    fn test_async_cluster_move_error_when_new_node_is_added() {
        let name = "rebuild_with_extra_nodes";

        let requests = atomic::AtomicUsize::new(0);
        let started = atomic::AtomicBool::new(false);
        let refreshed_map = HashMap::from([
            (6379, atomic::AtomicBool::new(false)),
            (6380, atomic::AtomicBool::new(false)),
        ]);

        let MockEnv {
            runtime,
            async_connection: mut connection,
            handler: _handler,
            ..
        } = MockEnv::new(name, move |cmd: &[u8], port| {
            if !started.load(atomic::Ordering::SeqCst) {
                respond_startup(name, cmd)?;
            }
            started.store(true, atomic::Ordering::SeqCst);

            if contains_slice(cmd, b"PING") || contains_slice(cmd, b"SETNAME") {
                return Err(Ok(Value::SimpleString("OK".into())));
            }

            let i = requests.fetch_add(1, atomic::Ordering::SeqCst);

            let is_get_cmd = contains_slice(cmd, b"GET");
            let get_response = Err(Ok(Value::BulkString(b"123".to_vec())));
            match i {
                // Respond that the key exists on a node that does not yet have a connection:
                0 => Err(parse_redis_value(
                    format!("-MOVED 123 {name}:6380\r\n").as_bytes(),
                )),
                _ => {
                    if contains_slice(cmd, b"CLUSTER") && contains_slice(cmd, b"SLOTS") {
                        // Should not attempt to refresh slots more than once,
                        // so we expect a single CLUSTER NODES request for each node
                        assert!(!refreshed_map
                            .get(&port)
                            .unwrap()
                            .swap(true, Ordering::SeqCst));
                        Err(Ok(Value::Array(vec![
                            Value::Array(vec![
                                Value::Int(0),
                                Value::Int(1),
                                Value::Array(vec![
                                    Value::BulkString(name.as_bytes().to_vec()),
                                    Value::Int(6379),
                                ]),
                            ]),
                            Value::Array(vec![
                                Value::Int(2),
                                Value::Int(16383),
                                Value::Array(vec![
                                    Value::BulkString(name.as_bytes().to_vec()),
                                    Value::Int(6380),
                                ]),
                            ]),
                        ])))
                    } else {
                        assert_eq!(port, 6380);
                        assert!(is_get_cmd, "{:?}", std::str::from_utf8(cmd));
                        get_response
                    }
                }
            }
        });

        let value = runtime.block_on(
            cmd("GET")
                .arg("test")
                .query_async::<_, Option<i32>>(&mut connection),
        );

        assert_eq!(value, Ok(Some(123)));
    }

    fn test_async_cluster_refresh_topology_after_moved_assert_get_succeed_and_expected_retries(
        slots_config_vec: Vec<Vec<MockSlotRange>>,
        ports: Vec<u16>,
        has_a_majority: bool,
    ) {
        assert!(!ports.is_empty() && !slots_config_vec.is_empty());
        let name = "refresh_topology_moved";
        let num_of_nodes = ports.len();
        let requests = atomic::AtomicUsize::new(0);
        let started = atomic::AtomicBool::new(false);
        let refresh_calls = Arc::new(atomic::AtomicUsize::new(0));
        let refresh_calls_cloned = refresh_calls.clone();
        let MockEnv {
            runtime,
            async_connection: mut connection,
            ..
        } = MockEnv::with_client_builder(
            ClusterClient::builder(vec![&*format!("redis://{name}")])
                // Disable the rate limiter to refresh slots immediately on all MOVED errors.
                .slots_refresh_rate_limit(Duration::from_secs(0), 0),
            name,
            move |cmd: &[u8], port| {
                if !started.load(atomic::Ordering::SeqCst) {
                    respond_startup_with_replica_using_config(
                        name,
                        cmd,
                        Some(slots_config_vec[0].clone()),
                    )?;
                }
                started.store(true, atomic::Ordering::SeqCst);

                if contains_slice(cmd, b"PING") || contains_slice(cmd, b"SETNAME") {
                    return Err(Ok(Value::SimpleString("OK".into())));
                }

                let i = requests.fetch_add(1, atomic::Ordering::SeqCst);
                let is_get_cmd = contains_slice(cmd, b"GET");
                let get_response = Err(Ok(Value::BulkString(b"123".to_vec())));
                let moved_node = ports[0];
                match i {
                    // Respond that the key exists on a node that does not yet have a connection:
                    0 => Err(parse_redis_value(
                        format!("-MOVED 123 {name}:{moved_node}\r\n").as_bytes(),
                    )),
                    _ => {
                        if contains_slice(cmd, b"CLUSTER") && contains_slice(cmd, b"SLOTS") {
                            refresh_calls_cloned.fetch_add(1, atomic::Ordering::SeqCst);
                            let view_index =
                                get_node_view_index(slots_config_vec.len(), &ports, port);
                            Err(Ok(create_topology_from_config(
                                name,
                                slots_config_vec[view_index].clone(),
                            )))
                        } else {
                            assert_eq!(port, moved_node);
                            assert!(is_get_cmd, "{:?}", std::str::from_utf8(cmd));
                            get_response
                        }
                    }
                }
            },
        );
        runtime.block_on(async move {
        let res = cmd("GET")
            .arg("test")
            .query_async::<_, Option<i32>>(&mut connection)
            .await;
        assert_eq!(res, Ok(Some(123)));
        // If there is a majority in the topology views, or if it's a 2-nodes cluster, we shall be able to calculate the topology on the first try,
        // so each node will be queried only once with CLUSTER SLOTS.
        // Otherwise, if we don't have a majority, we expect to see the refresh_slots function being called with the maximum retry number.
        let expected_calls = if has_a_majority || num_of_nodes == 2 {num_of_nodes} else {DEFAULT_NUMBER_OF_REFRESH_SLOTS_RETRIES * num_of_nodes};
        let mut refreshed_calls = 0;
        for _ in 0..100 {
            refreshed_calls = refresh_calls.load(atomic::Ordering::Relaxed);
            if refreshed_calls == expected_calls {
                return;
            } else {
                let sleep_duration = core::time::Duration::from_millis(100);
                #[cfg(feature = "tokio-comp")]
                tokio::time::sleep(sleep_duration).await;
            }
        }
        panic!("Failed to reach to the expected topology refresh retries. Found={refreshed_calls}, Expected={expected_calls}")
    });
    }

    fn test_async_cluster_refresh_slots_rate_limiter_helper(
        slots_config_vec: Vec<Vec<MockSlotRange>>,
        ports: Vec<u16>,
        should_skip: bool,
    ) {
        // This test queries GET, which returns a MOVED error. If `should_skip` is true,
        // it indicates that we should skip refreshing slots because the specified time
        // duration since the last refresh slots call has not yet passed. In this case,
        // we expect CLUSTER SLOTS not to be called on the nodes after receiving the
        // MOVED error.

        // If `should_skip` is false, we verify that if the MOVED error occurs after the
        // time duration of the rate limiter has passed, the refresh slots operation
        // should not be skipped. We assert this by expecting calls to CLUSTER SLOTS on
        // all nodes.
        let test_name = format!(
            "test_async_cluster_refresh_slots_rate_limiter_helper_{}",
            if should_skip {
                "should_skip"
            } else {
                "not_skipping_waiting_time_passed"
            }
        );

        let requests = atomic::AtomicUsize::new(0);
        let started = atomic::AtomicBool::new(false);
        let refresh_calls = Arc::new(atomic::AtomicUsize::new(0));
        let refresh_calls_cloned = Arc::clone(&refresh_calls);
        let wait_duration = Duration::from_millis(10);
        let num_of_nodes = ports.len();

        let MockEnv {
            runtime,
            async_connection: mut connection,
            ..
        } = MockEnv::with_client_builder(
            ClusterClient::builder(vec![&*format!("redis://{test_name}")])
                .slots_refresh_rate_limit(wait_duration, 0),
            test_name.clone().as_str(),
            move |cmd: &[u8], port| {
                if !started.load(atomic::Ordering::SeqCst) {
                    respond_startup_with_replica_using_config(
                        test_name.as_str(),
                        cmd,
                        Some(slots_config_vec[0].clone()),
                    )?;
                    started.store(true, atomic::Ordering::SeqCst);
                }

                if contains_slice(cmd, b"PING") {
                    return Err(Ok(Value::SimpleString("OK".into())));
                }

                let i = requests.fetch_add(1, atomic::Ordering::SeqCst);
                let is_get_cmd = contains_slice(cmd, b"GET");
                let get_response = Err(Ok(Value::BulkString(b"123".to_vec())));
                let moved_node = ports[0];
                match i {
                    // The first request calls are the starting calls for each GET command where we want to respond with MOVED error
                    0 => {
                        if !should_skip {
                            // Wait for the wait duration to pass
                            std::thread::sleep(wait_duration.add(Duration::from_millis(10)));
                        }
                        Err(parse_redis_value(
                            format!("-MOVED 123 {test_name}:{moved_node}\r\n").as_bytes(),
                        ))
                    }
                    _ => {
                        if contains_slice(cmd, b"CLUSTER") && contains_slice(cmd, b"SLOTS") {
                            refresh_calls_cloned.fetch_add(1, atomic::Ordering::SeqCst);
                            let view_index =
                                get_node_view_index(slots_config_vec.len(), &ports, port);
                            Err(Ok(create_topology_from_config(
                                test_name.as_str(),
                                slots_config_vec[view_index].clone(),
                            )))
                        } else {
                            // Even if the slots weren't refreshed we still expect the command to be
                            // routed by the redirect host and port it received in the moved error
                            assert_eq!(port, moved_node);
                            assert!(is_get_cmd, "{:?}", std::str::from_utf8(cmd));
                            get_response
                        }
                    }
                }
            },
        );

        runtime.block_on(async move {
            // First GET request should raise MOVED error and then refresh slots
            let res = cmd("GET")
                .arg("test")
                .query_async::<_, Option<i32>>(&mut connection)
                .await;
            assert_eq!(res, Ok(Some(123)));

            // We should skip is false, we should call CLUSTER SLOTS once per node
            let expected_calls = if should_skip {
                0
            } else {
                num_of_nodes
            };
            for _ in 0..4 {
                if refresh_calls.load(atomic::Ordering::Relaxed) == expected_calls {
                    return Ok::<_, RedisError>(());
                }
                let _ = sleep(Duration::from_millis(50).into()).await;
            }
            panic!("Refresh slots wasn't called as expected!\nExpected CLUSTER SLOTS calls: {}, actual calls: {:?}", expected_calls, refresh_calls.load(atomic::Ordering::Relaxed));
        }).unwrap()
    }

    fn test_async_cluster_refresh_topology_in_client_init_get_succeed(
        slots_config_vec: Vec<Vec<MockSlotRange>>,
        ports: Vec<u16>,
    ) {
        assert!(!ports.is_empty() && !slots_config_vec.is_empty());
        let name = "refresh_topology_client_init";
        let started = atomic::AtomicBool::new(false);
        let MockEnv {
            runtime,
            async_connection: mut connection,
            ..
        } = MockEnv::with_client_builder(
            ClusterClient::builder::<String>(
                ports
                    .iter()
                    .map(|port| format!("redis://{name}:{port}"))
                    .collect::<Vec<_>>(),
            ),
            name,
            move |cmd: &[u8], port| {
                let is_started = started.load(atomic::Ordering::SeqCst);
                if !is_started {
                    if contains_slice(cmd, b"PING") || contains_slice(cmd, b"SETNAME") {
                        return Err(Ok(Value::SimpleString("OK".into())));
                    } else if contains_slice(cmd, b"CLUSTER") && contains_slice(cmd, b"SLOTS") {
                        let view_index = get_node_view_index(slots_config_vec.len(), &ports, port);
                        return Err(Ok(create_topology_from_config(
                            name,
                            slots_config_vec[view_index].clone(),
                        )));
                    } else if contains_slice(cmd, b"READONLY") {
                        return Err(Ok(Value::SimpleString("OK".into())));
                    }
                }
                started.store(true, atomic::Ordering::SeqCst);
                if contains_slice(cmd, b"PING") {
                    return Err(Ok(Value::SimpleString("OK".into())));
                }

                let is_get_cmd = contains_slice(cmd, b"GET");
                let get_response = Err(Ok(Value::BulkString(b"123".to_vec())));
                {
                    assert!(is_get_cmd, "{:?}", std::str::from_utf8(cmd));
                    get_response
                }
            },
        );
        let value = runtime.block_on(
            cmd("GET")
                .arg("test")
                .query_async::<_, Option<i32>>(&mut connection),
        );

        assert_eq!(value, Ok(Some(123)));
    }

    fn generate_topology_view(
        ports: &[u16],
        interval: usize,
        full_slot_coverage: bool,
    ) -> Vec<MockSlotRange> {
        let mut slots_res = vec![];
        let mut start_pos: usize = 0;
        for (idx, port) in ports.iter().enumerate() {
            let end_pos: usize = if idx == ports.len() - 1 && full_slot_coverage {
                16383
            } else {
                start_pos + interval
            };
            let mock_slot = MockSlotRange {
                primary_port: *port,
                replica_ports: vec![],
                slot_range: (start_pos as u16..end_pos as u16),
            };
            slots_res.push(mock_slot);
            start_pos = end_pos + 1;
        }
        slots_res
    }

    fn get_ports(num_of_nodes: usize) -> Vec<u16> {
        (6379_u16..6379 + num_of_nodes as u16).collect()
    }

    fn get_no_majority_topology_view(ports: &[u16]) -> Vec<Vec<MockSlotRange>> {
        let mut result = vec![];
        let mut full_coverage = true;
        for i in 0..ports.len() {
            result.push(generate_topology_view(ports, i + 1, full_coverage));
            full_coverage = !full_coverage;
        }
        result
    }

    fn get_topology_with_majority(ports: &[u16]) -> Vec<Vec<MockSlotRange>> {
        let view: Vec<MockSlotRange> = generate_topology_view(ports, 10, true);
        let result: Vec<_> = ports.iter().map(|_| view.clone()).collect();
        result
    }

    #[test]
    #[serial_test::serial]
    fn test_async_cluster_refresh_topology_after_moved_error_all_nodes_agree_get_succeed() {
        let ports = get_ports(3);
        test_async_cluster_refresh_topology_after_moved_assert_get_succeed_and_expected_retries(
            get_topology_with_majority(&ports),
            ports,
            true,
        );
    }

    #[test]
    #[serial_test::serial]
    fn test_async_cluster_refresh_topology_in_client_init_all_nodes_agree_get_succeed() {
        let ports = get_ports(3);
        test_async_cluster_refresh_topology_in_client_init_get_succeed(
            get_topology_with_majority(&ports),
            ports,
        );
    }

    #[test]
    #[serial_test::serial]
    fn test_async_cluster_refresh_topology_after_moved_error_with_no_majority_get_succeed() {
        for num_of_nodes in 2..4 {
            let ports = get_ports(num_of_nodes);
            test_async_cluster_refresh_topology_after_moved_assert_get_succeed_and_expected_retries(
                get_no_majority_topology_view(&ports),
                ports,
                false,
            );
        }
    }

    #[test]
    #[serial_test::serial]
    fn test_async_cluster_refresh_topology_in_client_init_with_no_majority_get_succeed() {
        for num_of_nodes in 2..4 {
            let ports = get_ports(num_of_nodes);
            test_async_cluster_refresh_topology_in_client_init_get_succeed(
                get_no_majority_topology_view(&ports),
                ports,
            );
        }
    }

    #[test]
    #[serial_test::serial]
    fn test_async_cluster_refresh_topology_even_with_zero_retries() {
        let name = "test_async_cluster_refresh_topology_even_with_zero_retries";

        let should_refresh = atomic::AtomicBool::new(false);

        let MockEnv {
            runtime,
            async_connection: mut connection,
            handler: _handler,
            ..
        } = MockEnv::with_client_builder(
            ClusterClient::builder(vec![&*format!("redis://{name}")]).retries(0)
            // Disable the rate limiter to refresh slots immediately on the MOVED error.
            .slots_refresh_rate_limit(Duration::from_secs(0), 0),
            name,
            move |cmd: &[u8], port| {
                if !should_refresh.load(atomic::Ordering::SeqCst) {
                    respond_startup(name, cmd)?;
                }

                if contains_slice(cmd, b"PING") || contains_slice(cmd, b"SETNAME") {
                    return Err(Ok(Value::SimpleString("OK".into())));
                }

                if contains_slice(cmd, b"CLUSTER") && contains_slice(cmd, b"SLOTS") {
                    return Err(Ok(Value::Array(vec![
                        Value::Array(vec![
                            Value::Int(0),
                            Value::Int(1),
                            Value::Array(vec![
                                Value::BulkString(name.as_bytes().to_vec()),
                                Value::Int(6379),
                            ]),
                        ]),
                        Value::Array(vec![
                            Value::Int(2),
                            Value::Int(16383),
                            Value::Array(vec![
                                Value::BulkString(name.as_bytes().to_vec()),
                                Value::Int(6380),
                            ]),
                        ]),
                    ])));
                }

                if contains_slice(cmd, b"GET") {
                    let get_response = Err(Ok(Value::BulkString(b"123".to_vec())));
                    match port {
                        6380 => get_response,
                        // Respond that the key exists on a node that does not yet have a connection:
                        _ => {
                            // Should not attempt to refresh slots more than once:
                            assert!(!should_refresh.swap(true, Ordering::SeqCst));
                            Err(parse_redis_value(
                                format!("-MOVED 123 {name}:6380\r\n").as_bytes(),
                            ))
                        }
                    }
                } else {
                    panic!("unexpected command {cmd:?}")
                }
            },
        );

        let value = runtime.block_on(
            cmd("GET")
                .arg("test")
                .query_async::<_, Option<i32>>(&mut connection),
        );

        // The user should receive an initial error, because there are no retries and the first request failed.
        assert_eq!(
            value,
            Err(RedisError::from((
                ErrorKind::Moved,
                "An error was signalled by the server",
                "test_async_cluster_refresh_topology_even_with_zero_retries:6380".to_string()
            )))
        );

        let value = runtime.block_on(
            cmd("GET")
                .arg("test")
                .query_async::<_, Option<i32>>(&mut connection),
        );

        assert_eq!(value, Ok(Some(123)));
    }

    #[test]
    fn test_async_cluster_update_slots_based_on_moved_error_indicates_slot_migration() {
        // This test simulates the scenario where the client receives a MOVED error indicating that a key is now
        // stored on the primary node of another shard.
        // It ensures that the new slot now owned by the primary and its associated replicas.
        let name = "test_async_cluster_update_slots_based_on_moved_error_indicates_slot_migration";
        let slots_config = vec![
            MockSlotRange {
                primary_port: 6379,
                replica_ports: vec![7000],
                slot_range: (0..8000),
            },
            MockSlotRange {
                primary_port: 6380,
                replica_ports: vec![7001],
                slot_range: (8001..16380),
            },
        ];

        let moved_from_port = 6379;
        let moved_to_port = 6380;
        let new_shard_replica_port = 7001;

        // Tracking moved and replica requests for validation
        let moved_requests = Arc::new(atomic::AtomicUsize::new(0));
        let cloned_moved_requests = moved_requests.clone();
        let replica_requests = Arc::new(atomic::AtomicUsize::new(0));
        let cloned_replica_requests = moved_requests.clone();

        // Test key and slot
        let key = "test";
        let key_slot = 6918;

        // Mock environment setup
        let MockEnv {
            runtime,
            async_connection: mut connection,
            handler: _handler,
            ..
        } = MockEnv::with_client_builder(
            ClusterClient::builder(vec![&*format!("redis://{name}")])
                    .slots_refresh_rate_limit(Duration::from_secs(1000000), 0) // Rate limiter to disable slot refresh
                    .read_from_replicas(), // Allow reads from replicas
            name,
            move |cmd: &[u8], port| {
                if contains_slice(cmd, b"PING")
                    || contains_slice(cmd, b"SETNAME")
                    || contains_slice(cmd, b"READONLY")
                {
                    return Err(Ok(Value::SimpleString("OK".into())));
                }

                if contains_slice(cmd, b"CLUSTER") && contains_slice(cmd, b"SLOTS") {
                    let slots = create_topology_from_config(name, slots_config.clone());
                    return Err(Ok(slots));
                }

                if contains_slice(cmd, b"SET") {
                    if port == moved_to_port {
                        // Simulate primary OK response
                        Err(Ok(Value::SimpleString("OK".into())))
                    } else if port == moved_from_port {
                        // Simulate MOVED error for other port
                        moved_requests.fetch_add(1, Ordering::Relaxed);
                        Err(parse_redis_value(
                            format!("-MOVED {key_slot} {name}:{moved_to_port}\r\n").as_bytes(),
                        ))
                    } else {
                        panic!("unexpected port for SET command: {port:?}.\n
                            Expected one of: moved_to_port={moved_to_port}, moved_from_port={moved_from_port}");
                    }
                } else if contains_slice(cmd, b"GET") {
                    if new_shard_replica_port == port {
                        // Simulate replica response for GET after slot migration
                        replica_requests.fetch_add(1, Ordering::Relaxed);
                        Err(Ok(Value::BulkString(b"123".to_vec())))
                    } else {
                        panic!("unexpected port for GET command: {port:?}, Expected: {new_shard_replica_port:?}");
                    }
                } else {
                    panic!("unexpected command {cmd:?}")
                }
            },
        );

        // First request: Trigger MOVED error and reroute
        let value = runtime.block_on(
            cmd("SET")
                .arg(key)
                .arg("bar")
                .query_async::<_, Option<Value>>(&mut connection),
        );
        assert_eq!(value, Ok(Some(Value::SimpleString("OK".to_owned()))));

        // Second request: Should be routed directly to the new primary node if the slots map is updated
        let value = runtime.block_on(
            cmd("SET")
                .arg(key)
                .arg("bar")
                .query_async::<_, Option<Value>>(&mut connection),
        );
        assert_eq!(value, Ok(Some(Value::SimpleString("OK".to_owned()))));

        // Handle slot migration scenario: Ensure the new shard's replicas are accessible
        let value = runtime.block_on(
            cmd("GET")
                .arg(key)
                .query_async::<_, Option<i32>>(&mut connection),
        );
        assert_eq!(value, Ok(Some(123)));
        assert_eq!(cloned_replica_requests.load(Ordering::Relaxed), 1);

        // Assert there was only a single MOVED error
        assert_eq!(cloned_moved_requests.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_async_cluster_update_slots_based_on_moved_error_indicates_failover() {
        // This test simulates a failover scenario, where the client receives a MOVED error and the replica becomes the new primary.
        // The test verifies that the client updates the slot mapping to promote the replica to the primary and routes future requests
        // to the new primary, ensuring other slots in the shard are also handled by the new primary.
        let name = "test_async_cluster_update_slots_based_on_moved_error_indicates_failover";
        let slots_config = vec![
            MockSlotRange {
                primary_port: 6379,
                replica_ports: vec![7001],
                slot_range: (0..8000),
            },
            MockSlotRange {
                primary_port: 6380,
                replica_ports: vec![7002],
                slot_range: (8001..16380),
            },
        ];

        let moved_from_port = 6379;
        let moved_to_port = 7001;

        // Tracking moved for validation
        let moved_requests = Arc::new(atomic::AtomicUsize::new(0));
        let cloned_moved_requests = moved_requests.clone();

        // Test key and slot
        let key = "test";
        let key_slot = 6918;

        // Mock environment setup
        let MockEnv {
            runtime,
            async_connection: mut connection,
            handler: _handler,
            ..
        } = MockEnv::with_client_builder(
            ClusterClient::builder(vec![&*format!("redis://{name}")])
                .slots_refresh_rate_limit(Duration::from_secs(1000000), 0), // Rate limiter to disable slot refresh
            name,
            move |cmd: &[u8], port| {
                if contains_slice(cmd, b"PING")
                    || contains_slice(cmd, b"SETNAME")
                    || contains_slice(cmd, b"READONLY")
                {
                    return Err(Ok(Value::SimpleString("OK".into())));
                }

                if contains_slice(cmd, b"CLUSTER") && contains_slice(cmd, b"SLOTS") {
                    let slots = create_topology_from_config(name, slots_config.clone());
                    return Err(Ok(slots));
                }

                if contains_slice(cmd, b"SET") {
                    if port == moved_to_port {
                        // Simulate primary OK response
                        Err(Ok(Value::SimpleString("OK".into())))
                    } else if port == moved_from_port {
                        // Simulate MOVED error for other port
                        moved_requests.fetch_add(1, Ordering::Relaxed);
                        Err(parse_redis_value(
                            format!("-MOVED {key_slot} {name}:{moved_to_port}\r\n").as_bytes(),
                        ))
                    } else {
                        panic!("unexpected port for SET command: {port:?}.\n
                            Expected one of: moved_to_port={moved_to_port}, moved_from_port={moved_from_port}");
                    }
                } else {
                    panic!("unexpected command {cmd:?}")
                }
            },
        );

        // First request: Trigger MOVED error and reroute
        let value = runtime.block_on(
            cmd("SET")
                .arg(key)
                .arg("bar")
                .query_async::<_, Option<Value>>(&mut connection),
        );
        assert_eq!(value, Ok(Some(Value::SimpleString("OK".to_owned()))));

        // Second request: Should be routed directly to the new primary node if the slots map is updated
        let value = runtime.block_on(
            cmd("SET")
                .arg(key)
                .arg("bar")
                .query_async::<_, Option<Value>>(&mut connection),
        );
        assert_eq!(value, Ok(Some(Value::SimpleString("OK".to_owned()))));

        // Handle failover scenario: Ensure other slots in the same shard are updated to the new primary
        let key_slot_1044 = "foo2";
        let value = runtime.block_on(
            cmd("SET")
                .arg(key_slot_1044)
                .arg("bar2")
                .query_async::<_, Option<Value>>(&mut connection),
        );
        assert_eq!(value, Ok(Some(Value::SimpleString("OK".to_owned()))));

        // Assert there was only a single MOVED error
        assert_eq!(cloned_moved_requests.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_async_cluster_update_slots_based_on_moved_error_indicates_new_primary() {
        // This test simulates the scenario where the client receives a MOVED error indicating that the key now belongs to
        // an entirely new primary node that wasn't previously known. The test verifies that the client correctly adds the new
        // primary node to its slot map and routes future requests to the new node.
        let name = "test_async_cluster_update_slots_based_on_moved_error_indicates_new_primary";
        let slots_config = vec![
            MockSlotRange {
                primary_port: 6379,
                replica_ports: vec![],
                slot_range: (0..8000),
            },
            MockSlotRange {
                primary_port: 6380,
                replica_ports: vec![],
                slot_range: (8001..16380),
            },
        ];

        let moved_from_port = 6379;
        let moved_to_port = 6381;

        // Tracking moved for validation
        let moved_requests = Arc::new(atomic::AtomicUsize::new(0));
        let cloned_moved_requests = moved_requests.clone();

        // Test key and slot
        let key = "test";
        let key_slot = 6918;

        // Mock environment setup
        let MockEnv {
            runtime,
            async_connection: mut connection,
            handler: _handler,
            ..
        } = MockEnv::with_client_builder(
            ClusterClient::builder(vec![&*format!("redis://{name}")])
            .slots_refresh_rate_limit(Duration::from_secs(1000000), 0) // Rate limiter to disable slot refresh
            .read_from_replicas(), // Allow reads from replicas
            name,
            move |cmd: &[u8], port| {
                if contains_slice(cmd, b"PING")
                    || contains_slice(cmd, b"SETNAME")
                    || contains_slice(cmd, b"READONLY")
                {
                    return Err(Ok(Value::SimpleString("OK".into())));
                }

                if contains_slice(cmd, b"CLUSTER") && contains_slice(cmd, b"SLOTS") {
                    let slots = create_topology_from_config(name, slots_config.clone());
                    return Err(Ok(slots));
                }

                if contains_slice(cmd, b"SET") {
                    if port == moved_to_port {
                        // Simulate primary OK response
                        Err(Ok(Value::SimpleString("OK".into())))
                    } else if port == moved_from_port {
                        // Simulate MOVED error for other port
                        moved_requests.fetch_add(1, Ordering::Relaxed);
                        Err(parse_redis_value(
                            format!("-MOVED {key_slot} {name}:{moved_to_port}\r\n").as_bytes(),
                        ))
                    } else {
                        panic!("unexpected port for SET command: {port:?}.\n
                    Expected one of: moved_to_port={moved_to_port}, moved_from_port={moved_from_port}");
                    }
                } else if contains_slice(cmd, b"GET") {
                    if moved_to_port == port {
                        // Simulate primary response for GET
                        Err(Ok(Value::BulkString(b"123".to_vec())))
                    } else {
                        panic!(
                            "unexpected port for GET command: {port:?}, Expected: {moved_to_port}"
                        );
                    }
                } else {
                    panic!("unexpected command {cmd:?}")
                }
            },
        );

        // First request: Trigger MOVED error and reroute
        let value = runtime.block_on(
            cmd("SET")
                .arg(key)
                .arg("bar")
                .query_async::<_, Option<Value>>(&mut connection),
        );
        assert_eq!(value, Ok(Some(Value::SimpleString("OK".to_owned()))));

        // Second request: Should be routed directly to the new primary node if the slots map is updated
        let value = runtime.block_on(
            cmd("SET")
                .arg(key)
                .arg("bar")
                .query_async::<_, Option<Value>>(&mut connection),
        );
        assert_eq!(value, Ok(Some(Value::SimpleString("OK".to_owned()))));

        // Third request: The new primary should have no replicas so it should be directed to it
        let value = runtime.block_on(
            cmd("GET")
                .arg(key)
                .query_async::<_, Option<i32>>(&mut connection),
        );
        assert_eq!(value, Ok(Some(123)));

        // Assert there was only a single MOVED error
        assert_eq!(cloned_moved_requests.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_async_cluster_update_slots_based_on_moved_error_indicates_replica_of_different_shard() {
        // This test simulates a scenario where the client receives a MOVED error indicating that a key
        // has been moved to a replica in a different shard. The replica is then promoted to primary and
        // no longer exists in the shard’s replica set.
        // The test validates that the key gets correctly routed to the new primary and ensures that the
        // shard updates its mapping accordingly, with only one MOVED error encountered during the process.

        let name = "test_async_cluster_update_slots_based_on_moved_error_indicates_replica_of_different_shard";
        let slots_config = vec![
            MockSlotRange {
                primary_port: 6379,
                replica_ports: vec![7000],
                slot_range: (0..8000),
            },
            MockSlotRange {
                primary_port: 6380,
                replica_ports: vec![7001],
                slot_range: (8001..16380),
            },
        ];

        let moved_from_port = 6379;
        let moved_to_port = 7001;
        let primary_shard2 = 6380;

        // Tracking moved for validation
        let moved_requests = Arc::new(atomic::AtomicUsize::new(0));
        let cloned_moved_requests = moved_requests.clone();

        // Test key and slot of the first shard
        let key = "test";
        let key_slot = 6918;

        // Test key of the second shard
        let key_shard2 = "foo"; // slot 12182

        // Mock environment setup
        let MockEnv {
            runtime,
            async_connection: mut connection,
            handler: _handler,
            ..
        } = MockEnv::with_client_builder(
            ClusterClient::builder(vec![&*format!("redis://{name}")])
                    .slots_refresh_rate_limit(Duration::from_secs(1000000), 0) // Rate limiter to disable slot refresh
                    .read_from_replicas(), // Allow reads from replicas
            name,
            move |cmd: &[u8], port| {
                if contains_slice(cmd, b"PING")
                    || contains_slice(cmd, b"SETNAME")
                    || contains_slice(cmd, b"READONLY")
                {
                    return Err(Ok(Value::SimpleString("OK".into())));
                }

                if contains_slice(cmd, b"CLUSTER") && contains_slice(cmd, b"SLOTS") {
                    let slots = create_topology_from_config(name, slots_config.clone());
                    return Err(Ok(slots));
                }

                if contains_slice(cmd, b"SET") {
                    if port == moved_to_port {
                        // Simulate primary OK response
                        Err(Ok(Value::SimpleString("OK".into())))
                    } else if port == moved_from_port {
                        // Simulate MOVED error for other port
                        moved_requests.fetch_add(1, Ordering::Relaxed);
                        Err(parse_redis_value(
                            format!("-MOVED {key_slot} {name}:{moved_to_port}\r\n").as_bytes(),
                        ))
                    } else {
                        panic!("unexpected port for SET command: {port:?}.\n
                            Expected one of: moved_to_port={moved_to_port}, moved_from_port={moved_from_port}");
                    }
                } else if contains_slice(cmd, b"GET") {
                    if port == primary_shard2 {
                        // Simulate second shard primary response for GET
                        Err(Ok(Value::BulkString(b"123".to_vec())))
                    } else {
                        panic!("unexpected port for GET command: {port:?}, Expected: {primary_shard2:?}");
                    }
                } else {
                    panic!("unexpected command {cmd:?}")
                }
            },
        );

        // First request: Trigger MOVED error and reroute
        let value = runtime.block_on(
            cmd("SET")
                .arg(key)
                .arg("bar")
                .query_async::<_, Option<Value>>(&mut connection),
        );
        assert_eq!(value, Ok(Some(Value::SimpleString("OK".to_owned()))));

        // Second request: Should be routed directly to the new primary node if the slots map is updated
        let value = runtime.block_on(
            cmd("SET")
                .arg(key)
                .arg("bar")
                .query_async::<_, Option<Value>>(&mut connection),
        );
        assert_eq!(value, Ok(Some(Value::SimpleString("OK".to_owned()))));

        // Third request: Verify that the promoted replica is no longer part of the second shard replicas by
        // ensuring the response is received from the shard's primary
        let value = runtime.block_on(
            cmd("GET")
                .arg(key_shard2)
                .query_async::<_, Option<i32>>(&mut connection),
        );
        assert_eq!(value, Ok(Some(123)));

        // Assert there was only a single MOVED error
        assert_eq!(cloned_moved_requests.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_async_cluster_update_slots_based_on_moved_error_no_change() {
        // This test simulates a scenario where the client receives a MOVED error, but the new primary is the
        // same as the old primary (no actual change). It ensures that no additional slot map
        // updates are required and that the subsequent requests are still routed to the same primary node, with
        // only one MOVED error encountered.
        let name = "test_async_cluster_update_slots_based_on_moved_error_no_change";
        let slots_config = vec![
            MockSlotRange {
                primary_port: 6379,
                replica_ports: vec![7000],
                slot_range: (0..8000),
            },
            MockSlotRange {
                primary_port: 6380,
                replica_ports: vec![7001],
                slot_range: (8001..16380),
            },
        ];

        let moved_from_port = 6379;
        let moved_to_port = 6379;

        // Tracking moved for validation
        let moved_requests = Arc::new(atomic::AtomicUsize::new(0));
        let cloned_moved_requests = moved_requests.clone();

        // Test key and slot of the first shard
        let key = "test";
        let key_slot = 6918;

        // Mock environment setup
        let MockEnv {
            runtime,
            async_connection: mut connection,
            handler: _handler,
            ..
        } = MockEnv::with_client_builder(
            ClusterClient::builder(vec![&*format!("redis://{name}")])
                .slots_refresh_rate_limit(Duration::from_secs(1000000), 0), // Rate limiter to disable slot refresh
            name,
            move |cmd: &[u8], port| {
                if contains_slice(cmd, b"PING")
                    || contains_slice(cmd, b"SETNAME")
                    || contains_slice(cmd, b"READONLY")
                {
                    return Err(Ok(Value::SimpleString("OK".into())));
                }

                if contains_slice(cmd, b"CLUSTER") && contains_slice(cmd, b"SLOTS") {
                    let slots = create_topology_from_config(name, slots_config.clone());
                    return Err(Ok(slots));
                }

                if contains_slice(cmd, b"SET") {
                    if port == moved_to_port {
                        if moved_requests.load(Ordering::Relaxed) == 0 {
                            moved_requests.fetch_add(1, Ordering::Relaxed);
                            Err(parse_redis_value(
                                format!("-MOVED {key_slot} {name}:{moved_to_port}\r\n").as_bytes(),
                            ))
                        } else {
                            Err(Ok(Value::SimpleString("OK".into())))
                        }
                    } else {
                        panic!("unexpected port for SET command: {port:?}.\n
                            Expected one of: moved_to_port={moved_to_port}, moved_from_port={moved_from_port}");
                    }
                } else {
                    panic!("unexpected command {cmd:?}")
                }
            },
        );

        // First request: Trigger MOVED error and reroute
        let value = runtime.block_on(
            cmd("SET")
                .arg(key)
                .arg("bar")
                .query_async::<_, Option<Value>>(&mut connection),
        );
        assert_eq!(value, Ok(Some(Value::SimpleString("OK".to_owned()))));

        // Second request: Should be still routed to the same primary node
        let value = runtime.block_on(
            cmd("SET")
                .arg(key)
                .arg("bar")
                .query_async::<_, Option<Value>>(&mut connection),
        );
        assert_eq!(value, Ok(Some(Value::SimpleString("OK".to_owned()))));

        // Assert there was only a single MOVED error
        assert_eq!(cloned_moved_requests.load(Ordering::Relaxed), 1);
    }

    #[test]
    #[serial_test::serial]
    fn test_async_cluster_reconnect_even_with_zero_retries() {
        let name = "test_async_cluster_reconnect_even_with_zero_retries";

        let should_reconnect = atomic::AtomicBool::new(true);
        let connection_count = Arc::new(atomic::AtomicU16::new(0));
        let connection_count_clone = connection_count.clone();

        let MockEnv {
            runtime,
            async_connection: mut connection,
            handler: _handler,
            ..
        } = MockEnv::with_client_builder(
            ClusterClient::builder(vec![&*format!("redis://{name}")]).retries(0),
            name,
            move |cmd: &[u8], port| {
                match respond_startup(name, cmd) {
                    Ok(_) => {}
                    Err(err) => {
                        connection_count.fetch_add(1, Ordering::Relaxed);
                        return Err(err);
                    }
                }

                if contains_slice(cmd, b"ECHO") && port == 6379 {
                    // Should not attempt to refresh slots more than once:
                    if should_reconnect.swap(false, Ordering::SeqCst) {
                        Err(Err(broken_pipe_error()))
                    } else {
                        Err(Ok(Value::BulkString(b"PONG".to_vec())))
                    }
                } else {
                    panic!("unexpected command {cmd:?}")
                }
            },
        );

        // We expect 6 calls in total. MockEnv creates both synchronous and asynchronous connections, which make the following calls:
        // - 1 call by the sync connection to `CLUSTER SLOTS` for initializing the client's topology map.
        // - 3 calls by the async connection to `PING`: one for the user connection when creating the node from initial addresses,
        //     and two more for checking the user and management connections during client initialization in `refresh_slots`.
        // - 1 call by the async connection to `CLIENT SETNAME` for setting up the management connection name.
        // - 1 call by the async connection to `CLUSTER SLOTS` for initializing the client's topology map.
        // Note: If additional nodes or setup calls are added, this number should increase.
        let expected_init_calls = 6;
        assert_eq!(
            connection_count_clone.load(Ordering::Relaxed),
            expected_init_calls
        );

        let value = runtime.block_on(connection.route_command(
            &cmd("ECHO"),
            RoutingInfo::SingleNode(SingleNodeRoutingInfo::ByAddress {
                host: name.to_string(),
                port: 6379,
            }),
        ));

        // The user should receive an initial error, because there are no retries and the first request failed.
        assert_eq!(
            value.unwrap_err().to_string(),
            broken_pipe_error().to_string()
        );

        let value = runtime.block_on(connection.route_command(
            &cmd("ECHO"),
            RoutingInfo::SingleNode(SingleNodeRoutingInfo::ByAddress {
                host: name.to_string(),
                port: 6379,
            }),
        ));

        assert_eq!(value, Ok(Value::BulkString(b"PONG".to_vec())));
        // `expected_init_calls` plus another PING for a new user connection created from refresh_connections
        assert_eq!(
            connection_count_clone.load(Ordering::Relaxed),
            expected_init_calls + 1
        );
    }

    #[test]
    #[serial_test::serial]
    fn test_async_cluster_refresh_slots_rate_limiter_skips_refresh() {
        let ports = get_ports(3);
        test_async_cluster_refresh_slots_rate_limiter_helper(
            get_topology_with_majority(&ports),
            ports,
            true,
        );
    }

    #[test]
    #[serial_test::serial]
    fn test_async_cluster_refresh_slots_rate_limiter_does_refresh_when_wait_duration_passed() {
        let ports = get_ports(3);
        test_async_cluster_refresh_slots_rate_limiter_helper(
            get_topology_with_majority(&ports),
            ports,
            false,
        );
    }

    #[test]
    #[serial_test::serial]
    fn test_async_cluster_ask_redirect() {
        let name = "node";
        let completed = Arc::new(AtomicI32::new(0));
        let MockEnv {
            async_connection: mut connection,
            handler: _handler,
            runtime,
            ..
        } = MockEnv::with_client_builder(
            ClusterClient::builder(vec![&*format!("redis://{name}")]),
            name,
            {
                move |cmd: &[u8], port| {
                    respond_startup_two_nodes(name, cmd)?;
                    // Error twice with io-error, ensure connection is reestablished w/out calling
                    // other node (i.e., not doing a full slot rebuild)
                    let count = completed.fetch_add(1, Ordering::SeqCst);
                    match port {
                        6379 => match count {
                            0 => Err(parse_redis_value(b"-ASK 14000 node:6380\r\n")),
                            _ => panic!("Node should not be called now"),
                        },
                        6380 => match count {
                            1 => {
                                assert!(contains_slice(cmd, b"ASKING"));
                                Err(Ok(Value::Okay))
                            }
                            2 => {
                                assert!(contains_slice(cmd, b"GET"));
                                Err(Ok(Value::BulkString(b"123".to_vec())))
                            }
                            _ => panic!("Node should not be called now"),
                        },
                        _ => panic!("Wrong node"),
                    }
                }
            },
        );

        let value = runtime.block_on(
            cmd("GET")
                .arg("test")
                .query_async::<_, Option<i32>>(&mut connection),
        );

        assert_eq!(value, Ok(Some(123)));
    }

    #[test]
    #[serial_test::serial]
    fn test_async_cluster_ask_save_new_connection() {
        let name = "node";
        let ping_attempts = Arc::new(AtomicI32::new(0));
        let ping_attempts_clone = ping_attempts.clone();
        let MockEnv {
            async_connection: mut connection,
            handler: _handler,
            runtime,
            ..
        } = MockEnv::with_client_builder(
            ClusterClient::builder(vec![&*format!("redis://{name}")]),
            name,
            {
                move |cmd: &[u8], port| {
                    if port != 6391 {
                        respond_startup_two_nodes(name, cmd)?;
                        return Err(parse_redis_value(b"-ASK 14000 node:6391\r\n"));
                    }

                    if contains_slice(cmd, b"PING") {
                        ping_attempts_clone.fetch_add(1, Ordering::Relaxed);
                    }
                    respond_startup_two_nodes(name, cmd)?;
                    Err(Ok(Value::Okay))
                }
            },
        );

        for _ in 0..4 {
            runtime
                .block_on(
                    cmd("GET")
                        .arg("test")
                        .query_async::<_, Value>(&mut connection),
                )
                .unwrap();
        }

        assert_eq!(ping_attempts.load(Ordering::Relaxed), 1);
    }

    #[test]
    #[serial_test::serial]
    fn test_async_cluster_reset_routing_if_redirect_fails() {
        let name = "test_async_cluster_reset_routing_if_redirect_fails";
        let completed = Arc::new(AtomicI32::new(0));
        let MockEnv {
            async_connection: mut connection,
            handler: _handler,
            runtime,
            ..
        } = MockEnv::new(name, move |cmd: &[u8], port| {
            if port != 6379 && port != 6380 {
                return Err(Err(broken_pipe_error()));
            }
            respond_startup_two_nodes(name, cmd)?;
            let count = completed.fetch_add(1, Ordering::SeqCst);
            match (port, count) {
                // redirect once to non-existing node
                (6379, 0) => Err(parse_redis_value(
                    format!("-ASK 14000 {name}:9999\r\n").as_bytes(),
                )),
                // accept the next request
                (6379, 1) => {
                    assert!(contains_slice(cmd, b"GET"));
                    Err(Ok(Value::BulkString(b"123".to_vec())))
                }
                _ => panic!("Wrong node. port: {port}, received count: {count}"),
            }
        });

        let value = runtime.block_on(
            cmd("GET")
                .arg("test")
                .query_async::<_, Option<i32>>(&mut connection),
        );

        assert_eq!(value, Ok(Some(123)));
    }

    #[test]
    #[serial_test::serial]
    fn test_async_cluster_ask_redirect_even_if_original_call_had_no_route() {
        let name = "node";
        let completed = Arc::new(AtomicI32::new(0));
        let MockEnv {
            async_connection: mut connection,
            handler: _handler,
            runtime,
            ..
        } = MockEnv::with_client_builder(
            ClusterClient::builder(vec![&*format!("redis://{name}")]),
            name,
            {
                move |cmd: &[u8], port| {
                    respond_startup_two_nodes(name, cmd)?;
                    // Error twice with io-error, ensure connection is reestablished w/out calling
                    // other node (i.e., not doing a full slot rebuild)
                    let count = completed.fetch_add(1, Ordering::SeqCst);
                    if count == 0 {
                        return Err(parse_redis_value(b"-ASK 14000 node:6380\r\n"));
                    }
                    match port {
                        6380 => match count {
                            1 => {
                                assert!(
                                    contains_slice(cmd, b"ASKING"),
                                    "{:?}",
                                    std::str::from_utf8(cmd)
                                );
                                Err(Ok(Value::Okay))
                            }
                            2 => {
                                assert!(contains_slice(cmd, b"EVAL"));
                                Err(Ok(Value::Okay))
                            }
                            _ => panic!("Node should not be called now"),
                        },
                        _ => panic!("Wrong node"),
                    }
                }
            },
        );

        let value = runtime.block_on(
            cmd("EVAL") // Eval command has no directed, and so is redirected randomly
                .query_async::<_, Value>(&mut connection),
        );

        assert_eq!(value, Ok(Value::Okay));
    }

    #[test]
    #[serial_test::serial]
    fn test_async_cluster_ask_error_when_new_node_is_added() {
        let name = "ask_with_extra_nodes";

        let requests = atomic::AtomicUsize::new(0);
        let started = atomic::AtomicBool::new(false);

        let MockEnv {
            runtime,
            async_connection: mut connection,
            handler: _handler,
            ..
        } = MockEnv::new(name, move |cmd: &[u8], port| {
            if !started.load(atomic::Ordering::SeqCst) {
                respond_startup(name, cmd)?;
            }
            started.store(true, atomic::Ordering::SeqCst);

            if contains_slice(cmd, b"PING") || contains_slice(cmd, b"SETNAME") {
                return Err(Ok(Value::SimpleString("OK".into())));
            }

            let i = requests.fetch_add(1, atomic::Ordering::SeqCst);

            match i {
                // Respond that the key exists on a node that does not yet have a connection:
                0 => Err(parse_redis_value(
                    format!("-ASK 123 {name}:6380\r\n").as_bytes(),
                )),
                1 => {
                    assert_eq!(port, 6380);
                    assert!(contains_slice(cmd, b"ASKING"));
                    Err(Ok(Value::Okay))
                }
                2 => {
                    assert_eq!(port, 6380);
                    assert!(contains_slice(cmd, b"GET"));
                    Err(Ok(Value::BulkString(b"123".to_vec())))
                }
                _ => {
                    panic!("Unexpected request: {:?}", cmd);
                }
            }
        });

        let value = runtime.block_on(
            cmd("GET")
                .arg("test")
                .query_async::<_, Option<i32>>(&mut connection),
        );

        assert_eq!(value, Ok(Some(123)));
    }

    #[test]
    #[serial_test::serial]
    fn test_async_cluster_replica_read() {
        let name = "node";

        // requests should route to replica
        let MockEnv {
            runtime,
            async_connection: mut connection,
            handler: _handler,
            ..
        } = MockEnv::with_client_builder(
            ClusterClient::builder(vec![&*format!("redis://{name}")])
                .retries(0)
                .read_from_replicas(),
            name,
            move |cmd: &[u8], port| {
                respond_startup_with_replica(name, cmd)?;
                match port {
                    6380 => Err(Ok(Value::BulkString(b"123".to_vec()))),
                    _ => panic!("Wrong node"),
                }
            },
        );

        let value = runtime.block_on(
            cmd("GET")
                .arg("test")
                .query_async::<_, Option<i32>>(&mut connection),
        );
        assert_eq!(value, Ok(Some(123)));

        // requests should route to primary
        let MockEnv {
            runtime,
            async_connection: mut connection,
            handler: _handler,
            ..
        } = MockEnv::with_client_builder(
            ClusterClient::builder(vec![&*format!("redis://{name}")])
                .retries(0)
                .read_from_replicas(),
            name,
            move |cmd: &[u8], port| {
                respond_startup_with_replica(name, cmd)?;
                match port {
                    6379 => Err(Ok(Value::SimpleString("OK".into()))),
                    _ => panic!("Wrong node"),
                }
            },
        );

        let value = runtime.block_on(
            cmd("SET")
                .arg("test")
                .arg("123")
                .query_async::<_, Option<Value>>(&mut connection),
        );
        assert_eq!(value, Ok(Some(Value::SimpleString("OK".to_owned()))));
    }

    fn test_async_cluster_fan_out(
        command: &'static str,
        expected_ports: Vec<u16>,
        slots_config: Option<Vec<MockSlotRange>>,
    ) {
        let name = "node";
        let found_ports = Arc::new(std::sync::Mutex::new(Vec::new()));
        let ports_clone = found_ports.clone();
        let mut cmd = Cmd::new();
        for arg in command.split_whitespace() {
            cmd.arg(arg);
        }
        let packed_cmd = cmd.get_packed_command();
        // requests should route to replica
        let MockEnv {
            runtime,
            async_connection: mut connection,
            handler: _handler,
            ..
        } = MockEnv::with_client_builder(
            ClusterClient::builder(vec![&*format!("redis://{name}")])
                .retries(0)
                .read_from_replicas(),
            name,
            move |received_cmd: &[u8], port| {
                respond_startup_with_replica_using_config(
                    name,
                    received_cmd,
                    slots_config.clone(),
                )?;
                if received_cmd == packed_cmd {
                    ports_clone.lock().unwrap().push(port);
                    return Err(Ok(Value::SimpleString("OK".into())));
                }
                Ok(())
            },
        );

        let _ = runtime.block_on(cmd.query_async::<_, Option<()>>(&mut connection));
        found_ports.lock().unwrap().sort();
        // MockEnv creates 2 mock connections.
        assert_eq!(*found_ports.lock().unwrap(), expected_ports);
    }

    #[test]
    #[serial_test::serial]
    fn test_async_cluster_fan_out_to_all_primaries() {
        test_async_cluster_fan_out("FLUSHALL", vec![6379, 6381], None);
    }

    #[test]
    #[serial_test::serial]
    fn test_async_cluster_fan_out_to_all_nodes() {
        test_async_cluster_fan_out("CONFIG SET", vec![6379, 6380, 6381, 6382], None);
    }

    #[test]
    #[serial_test::serial]
    fn test_async_cluster_fan_out_once_to_each_primary_when_no_replicas_are_available() {
        test_async_cluster_fan_out(
            "CONFIG SET",
            vec![6379, 6381],
            Some(vec![
                MockSlotRange {
                    primary_port: 6379,
                    replica_ports: Vec::new(),
                    slot_range: (0..8191),
                },
                MockSlotRange {
                    primary_port: 6381,
                    replica_ports: Vec::new(),
                    slot_range: (8192..16383),
                },
            ]),
        );
    }

    #[test]
    #[serial_test::serial]
    fn test_async_cluster_fan_out_once_even_if_primary_has_multiple_slot_ranges() {
        test_async_cluster_fan_out(
            "CONFIG SET",
            vec![6379, 6380, 6381, 6382],
            Some(vec![
                MockSlotRange {
                    primary_port: 6379,
                    replica_ports: vec![6380],
                    slot_range: (0..4000),
                },
                MockSlotRange {
                    primary_port: 6381,
                    replica_ports: vec![6382],
                    slot_range: (4001..8191),
                },
                MockSlotRange {
                    primary_port: 6379,
                    replica_ports: vec![6380],
                    slot_range: (8192..8200),
                },
                MockSlotRange {
                    primary_port: 6381,
                    replica_ports: vec![6382],
                    slot_range: (8201..16383),
                },
            ]),
        );
    }

    #[test]
    #[serial_test::serial]
    fn test_async_cluster_route_according_to_passed_argument() {
        let name = "test_async_cluster_route_according_to_passed_argument";

        let touched_ports = Arc::new(std::sync::Mutex::new(Vec::new()));
        let cloned_ports = touched_ports.clone();

        // requests should route to replica
        let MockEnv {
            runtime,
            async_connection: mut connection,
            handler: _handler,
            ..
        } = MockEnv::with_client_builder(
            ClusterClient::builder(vec![&*format!("redis://{name}")])
                .retries(0)
                .read_from_replicas(),
            name,
            move |cmd: &[u8], port| {
                respond_startup_with_replica(name, cmd)?;
                cloned_ports.lock().unwrap().push(port);
                Err(Ok(Value::Nil))
            },
        );

        let mut cmd = cmd("GET");
        cmd.arg("test");
        let _ = runtime.block_on(connection.route_command(
            &cmd,
            RoutingInfo::MultiNode((MultipleNodeRoutingInfo::AllMasters, None)),
        ));
        {
            let mut touched_ports = touched_ports.lock().unwrap();
            touched_ports.sort();
            assert_eq!(*touched_ports, vec![6379, 6381]);
            touched_ports.clear();
        }

        let _ = runtime.block_on(connection.route_command(
            &cmd,
            RoutingInfo::MultiNode((MultipleNodeRoutingInfo::AllNodes, None)),
        ));
        {
            let mut touched_ports = touched_ports.lock().unwrap();
            touched_ports.sort();
            assert_eq!(*touched_ports, vec![6379, 6380, 6381, 6382]);
            touched_ports.clear();
        }

        let _ = runtime.block_on(connection.route_command(
            &cmd,
            RoutingInfo::SingleNode(SingleNodeRoutingInfo::ByAddress {
                host: name.to_string(),
                port: 6382,
            }),
        ));
        {
            let mut touched_ports = touched_ports.lock().unwrap();
            touched_ports.sort();
            assert_eq!(*touched_ports, vec![6382]);
            touched_ports.clear();
        }
    }

    #[test]
    #[serial_test::serial]
    fn test_async_cluster_fan_out_and_aggregate_numeric_response_with_min() {
        let name = "test_async_cluster_fan_out_and_aggregate_numeric_response";
        let mut cmd = Cmd::new();
        cmd.arg("SLOWLOG").arg("LEN");

        let MockEnv {
            runtime,
            async_connection: mut connection,
            handler: _handler,
            ..
        } = MockEnv::with_client_builder(
            ClusterClient::builder(vec![&*format!("redis://{name}")])
                .retries(0)
                .read_from_replicas(),
            name,
            move |received_cmd: &[u8], port| {
                respond_startup_with_replica_using_config(name, received_cmd, None)?;

                let res = 6383 - port as i64;
                Err(Ok(Value::Int(res))) // this results in 1,2,3,4
            },
        );

        let result = runtime
            .block_on(cmd.query_async::<_, i64>(&mut connection))
            .unwrap();
        assert_eq!(result, 10, "{result}");
    }

    #[test]
    #[serial_test::serial]
    fn test_async_cluster_fan_out_and_aggregate_logical_array_response() {
        let name = "test_async_cluster_fan_out_and_aggregate_logical_array_response";
        let mut cmd = Cmd::new();
        cmd.arg("SCRIPT")
            .arg("EXISTS")
            .arg("foo")
            .arg("bar")
            .arg("baz")
            .arg("barvaz");

        let MockEnv {
            runtime,
            async_connection: mut connection,
            handler: _handler,
            ..
        } = MockEnv::with_client_builder(
            ClusterClient::builder(vec![&*format!("redis://{name}")])
                .retries(0)
                .read_from_replicas(),
            name,
            move |received_cmd: &[u8], port| {
                respond_startup_with_replica_using_config(name, received_cmd, None)?;

                if port == 6381 {
                    return Err(Ok(Value::Array(vec![
                        Value::Int(0),
                        Value::Int(0),
                        Value::Int(1),
                        Value::Int(1),
                    ])));
                } else if port == 6379 {
                    return Err(Ok(Value::Array(vec![
                        Value::Int(0),
                        Value::Int(1),
                        Value::Int(0),
                        Value::Int(1),
                    ])));
                }

                panic!("unexpected port {port}");
            },
        );

        let result = runtime
            .block_on(cmd.query_async::<_, Vec<i64>>(&mut connection))
            .unwrap();
        assert_eq!(result, vec![0, 0, 0, 1], "{result:?}");
    }

    #[test]
    #[serial_test::serial]
    fn test_async_cluster_fan_out_and_return_one_succeeded_response() {
        let name = "test_async_cluster_fan_out_and_return_one_succeeded_response";
        let mut cmd = Cmd::new();
        cmd.arg("SCRIPT").arg("KILL");
        let MockEnv {
            runtime,
            async_connection: mut connection,
            handler: _handler,
            ..
        } = MockEnv::with_client_builder(
            ClusterClient::builder(vec![&*format!("redis://{name}")])
                .retries(0)
                .read_from_replicas(),
            name,
            move |received_cmd: &[u8], port| {
                respond_startup_with_replica_using_config(name, received_cmd, None)?;
                if port == 6381 {
                    return Err(Ok(Value::Okay));
                }
                Err(Err((
                    ErrorKind::NotBusy,
                    "No scripts in execution right now",
                )
                    .into()))
            },
        );

        let result = runtime
            .block_on(cmd.query_async::<_, Value>(&mut connection))
            .unwrap();
        assert_eq!(result, Value::Okay, "{result:?}");
    }

    #[test]
    #[serial_test::serial]
    fn test_async_cluster_fan_out_and_fail_one_succeeded_if_there_are_no_successes() {
        let name = "test_async_cluster_fan_out_and_fail_one_succeeded_if_there_are_no_successes";
        let mut cmd = Cmd::new();
        cmd.arg("SCRIPT").arg("KILL");
        let MockEnv {
            runtime,
            async_connection: mut connection,
            handler: _handler,
            ..
        } = MockEnv::with_client_builder(
            ClusterClient::builder(vec![&*format!("redis://{name}")])
                .retries(0)
                .read_from_replicas(),
            name,
            move |received_cmd: &[u8], _port| {
                respond_startup_with_replica_using_config(name, received_cmd, None)?;

                Err(Err((
                    ErrorKind::NotBusy,
                    "No scripts in execution right now",
                )
                    .into()))
            },
        );

        let result = runtime
            .block_on(cmd.query_async::<_, Value>(&mut connection))
            .unwrap_err();
        assert_eq!(result.kind(), ErrorKind::NotBusy, "{:?}", result.kind());
    }

    #[test]
    #[serial_test::serial]
    fn test_async_cluster_fan_out_and_return_all_succeeded_response() {
        let name = "test_async_cluster_fan_out_and_return_all_succeeded_response";
        let cmd = cmd("FLUSHALL");
        let MockEnv {
            runtime,
            async_connection: mut connection,
            handler: _handler,
            ..
        } = MockEnv::with_client_builder(
            ClusterClient::builder(vec![&*format!("redis://{name}")])
                .retries(0)
                .read_from_replicas(),
            name,
            move |received_cmd: &[u8], _port| {
                respond_startup_with_replica_using_config(name, received_cmd, None)?;
                Err(Ok(Value::Okay))
            },
        );

        let result = runtime
            .block_on(cmd.query_async::<_, Value>(&mut connection))
            .unwrap();
        assert_eq!(result, Value::Okay, "{result:?}");
    }

    #[test]
    #[serial_test::serial]
    fn test_async_cluster_fan_out_and_fail_all_succeeded_if_there_is_a_single_failure() {
        let name = "test_async_cluster_fan_out_and_fail_all_succeeded_if_there_is_a_single_failure";
        let cmd = cmd("FLUSHALL");
        let MockEnv {
            runtime,
            async_connection: mut connection,
            handler: _handler,
            ..
        } = MockEnv::with_client_builder(
            ClusterClient::builder(vec![&*format!("redis://{name}")])
                .retries(0)
                .read_from_replicas(),
            name,
            move |received_cmd: &[u8], port| {
                respond_startup_with_replica_using_config(name, received_cmd, None)?;
                if port == 6381 {
                    return Err(Err((
                        ErrorKind::NotBusy,
                        "No scripts in execution right now",
                    )
                        .into()));
                }
                Err(Ok(Value::Okay))
            },
        );

        let result = runtime
            .block_on(cmd.query_async::<_, Value>(&mut connection))
            .unwrap_err();
        assert_eq!(result.kind(), ErrorKind::NotBusy, "{:?}", result.kind());
    }

    #[test]
    #[serial_test::serial]
    fn test_async_cluster_first_succeeded_non_empty_or_all_empty_return_value_ignoring_nil_and_err_resps(
    ) {
        let name =
            "test_async_cluster_first_succeeded_non_empty_or_all_empty_return_value_ignoring_nil_and_err_resps";
        let cmd = cmd("RANDOMKEY");
        let MockEnv {
            runtime,
            async_connection: mut connection,
            handler: _handler,
            ..
        } = MockEnv::with_client_builder(
            ClusterClient::builder(vec![&*format!("redis://{name}")])
                .retries(0)
                .read_from_replicas(),
            name,
            move |received_cmd: &[u8], port| {
                let ports = vec![6379, 6380, 6381];
                let slots_config_vec = generate_topology_view(&ports, 1000, true);
                respond_startup_with_config(name, received_cmd, Some(slots_config_vec), false)?;
                if port == 6380 {
                    return Err(Ok(Value::BulkString("foo".as_bytes().to_vec())));
                } else if port == 6381 {
                    return Err(Err(RedisError::from((
                        redis::ErrorKind::ResponseError,
                        "ERROR",
                    ))));
                }
                Err(Ok(Value::Nil))
            },
        );

        let result = runtime
            .block_on(cmd.query_async::<_, String>(&mut connection))
            .unwrap();
        assert_eq!(result, "foo", "{result:?}");
    }

    #[test]
    #[serial_test::serial]
    fn test_async_cluster_first_succeeded_non_empty_or_all_empty_return_err_if_all_resps_are_nil_and_errors(
    ) {
        let name =
            "test_async_cluster_first_succeeded_non_empty_or_all_empty_return_err_if_all_resps_are_nil_and_errors";
        let cmd = cmd("RANDOMKEY");
        let MockEnv {
            runtime,
            async_connection: mut connection,
            handler: _handler,
            ..
        } = MockEnv::with_client_builder(
            ClusterClient::builder(vec![&*format!("redis://{name}")])
                .retries(0)
                .read_from_replicas(),
            name,
            move |received_cmd: &[u8], port| {
                respond_startup_with_config(name, received_cmd, None, false)?;
                if port == 6380 {
                    return Err(Ok(Value::Nil));
                }
                Err(Err(RedisError::from((
                    redis::ErrorKind::ResponseError,
                    "ERROR",
                ))))
            },
        );
        let result = runtime
            .block_on(cmd.query_async::<_, Value>(&mut connection))
            .unwrap_err();
        assert_eq!(result.kind(), ErrorKind::ResponseError);
    }

    #[test]
    #[serial_test::serial]
    fn test_async_cluster_first_succeeded_non_empty_or_all_empty_return_nil_if_all_resp_nil() {
        let name =
            "test_async_cluster_first_succeeded_non_empty_or_all_empty_return_nil_if_all_resp_nil";
        let cmd = cmd("RANDOMKEY");
        let MockEnv {
            runtime,
            async_connection: mut connection,
            handler: _handler,
            ..
        } = MockEnv::with_client_builder(
            ClusterClient::builder(vec![&*format!("redis://{name}")])
                .retries(0)
                .read_from_replicas(),
            name,
            move |received_cmd: &[u8], _port| {
                respond_startup_with_config(name, received_cmd, None, false)?;
                Err(Ok(Value::Nil))
            },
        );

        let result = runtime
            .block_on(cmd.query_async::<_, Value>(&mut connection))
            .unwrap();
        assert_eq!(result, Value::Nil, "{result:?}");
    }

    #[test]
    #[serial_test::serial]
    fn test_async_cluster_fan_out_and_return_map_of_results_for_special_response_policy() {
        let name = "foo";
        let mut cmd = Cmd::new();
        cmd.arg("LATENCY").arg("LATEST");
        let MockEnv {
            runtime,
            async_connection: mut connection,
            handler: _handler,
            ..
        } = MockEnv::with_client_builder(
            ClusterClient::builder(vec![&*format!("redis://{name}")])
                .retries(0)
                .read_from_replicas(),
            name,
            move |received_cmd: &[u8], port| {
                respond_startup_with_replica_using_config(name, received_cmd, None)?;
                Err(Ok(Value::BulkString(
                    format!("latency: {port}").into_bytes(),
                )))
            },
        );

        // TODO once RESP3 is in, return this as a map
        let mut result = runtime
            .block_on(cmd.query_async::<_, Vec<(String, String)>>(&mut connection))
            .unwrap();
        result.sort();
        assert_eq!(
            result,
            vec![
                (format!("{name}:6379"), "latency: 6379".to_string()),
                (format!("{name}:6380"), "latency: 6380".to_string()),
                (format!("{name}:6381"), "latency: 6381".to_string()),
                (format!("{name}:6382"), "latency: 6382".to_string())
            ],
            "{result:?}"
        );
    }

    #[test]
    #[serial_test::serial]
    fn test_async_cluster_fan_out_and_combine_arrays_of_values() {
        let name = "foo";
        let cmd = cmd("KEYS");
        let MockEnv {
            runtime,
            async_connection: mut connection,
            handler: _handler,
            ..
        } = MockEnv::with_client_builder(
            ClusterClient::builder(vec![&*format!("redis://{name}")])
                .retries(0)
                .read_from_replicas(),
            name,
            move |received_cmd: &[u8], port| {
                respond_startup_with_replica_using_config(name, received_cmd, None)?;
                Err(Ok(Value::Array(vec![Value::BulkString(
                    format!("key:{port}").into_bytes(),
                )])))
            },
        );

        let mut result = runtime
            .block_on(cmd.query_async::<_, Vec<String>>(&mut connection))
            .unwrap();
        result.sort();
        assert_eq!(
            result,
            vec!["key:6379".to_string(), "key:6381".to_string(),],
            "{result:?}"
        );
    }

    #[test]
    #[serial_test::serial]
    fn test_async_cluster_split_multi_shard_command_and_combine_arrays_of_values() {
        let name = "test_async_cluster_split_multi_shard_command_and_combine_arrays_of_values";
        let mut cmd = cmd("MGET");
        cmd.arg("foo").arg("bar").arg("baz");
        let MockEnv {
            runtime,
            async_connection: mut connection,
            handler: _handler,
            ..
        } = MockEnv::with_client_builder(
            ClusterClient::builder(vec![&*format!("redis://{name}")])
                .retries(0)
                .read_from_replicas(),
            name,
            move |received_cmd: &[u8], port| {
                respond_startup_with_replica_using_config(name, received_cmd, None)?;
                let cmd_str = std::str::from_utf8(received_cmd).unwrap();
                let results = ["foo", "bar", "baz"]
                    .iter()
                    .filter_map(|expected_key| {
                        if cmd_str.contains(expected_key) {
                            Some(Value::BulkString(
                                format!("{expected_key}-{port}").into_bytes(),
                            ))
                        } else {
                            None
                        }
                    })
                    .collect();
                Err(Ok(Value::Array(results)))
            },
        );

        let result = runtime
            .block_on(cmd.query_async::<_, Vec<String>>(&mut connection))
            .unwrap();
        assert_eq!(result, vec!["foo-6382", "bar-6380", "baz-6380"]);
    }

    #[test]
    #[serial_test::serial]
    fn test_async_cluster_handle_asking_error_in_split_multi_shard_command() {
        let name = "test_async_cluster_handle_asking_error_in_split_multi_shard_command";
        let mut cmd = cmd("MGET");
        cmd.arg("foo").arg("bar").arg("baz");
        let asking_called = Arc::new(AtomicU16::new(0));
        let asking_called_cloned = asking_called.clone();
        let MockEnv {
            runtime,
            async_connection: mut connection,
            handler: _handler,
            ..
        } = MockEnv::with_client_builder(
            ClusterClient::builder(vec![&*format!("redis://{name}")]).read_from_replicas(),
            name,
            move |received_cmd: &[u8], port| {
                respond_startup_with_replica_using_config(name, received_cmd, None)?;
                let cmd_str = std::str::from_utf8(received_cmd).unwrap();
                if cmd_str.contains("ASKING") && port == 6382 {
                    asking_called_cloned.fetch_add(1, Ordering::Relaxed);
                }
                if port == 6380 && cmd_str.contains("baz") {
                    return Err(parse_redis_value(
                        format!("-ASK 14000 {name}:6382\r\n").as_bytes(),
                    ));
                }
                let results = ["foo", "bar", "baz"]
                    .iter()
                    .filter_map(|expected_key| {
                        if cmd_str.contains(expected_key) {
                            Some(Value::BulkString(
                                format!("{expected_key}-{port}").into_bytes(),
                            ))
                        } else {
                            None
                        }
                    })
                    .collect();
                Err(Ok(Value::Array(results)))
            },
        );

        let result = runtime
            .block_on(cmd.query_async::<_, Vec<String>>(&mut connection))
            .unwrap();
        assert_eq!(result, vec!["foo-6382", "bar-6380", "baz-6382"]);
        assert_eq!(asking_called.load(Ordering::Relaxed), 1);
    }

    #[test]
    #[serial_test::serial]
    fn test_async_cluster_pass_errors_from_split_multi_shard_command() {
        let name = "test_async_cluster_pass_errors_from_split_multi_shard_command";
        let mut cmd = cmd("MGET");
        cmd.arg("foo").arg("bar").arg("baz");
        let MockEnv {
            runtime,
            async_connection: mut connection,
            ..
        } = MockEnv::new(name, move |received_cmd: &[u8], port| {
            respond_startup_with_replica_using_config(name, received_cmd, None)?;
            let cmd_str = std::str::from_utf8(received_cmd).unwrap();
            if cmd_str.contains("foo") || cmd_str.contains("baz") {
                Err(Err((ErrorKind::IoError, "error").into()))
            } else {
                Err(Ok(Value::Array(vec![Value::BulkString(
                    format!("{port}").into_bytes(),
                )])))
            }
        });

        let result = runtime
            .block_on(cmd.query_async::<_, Vec<String>>(&mut connection))
            .unwrap_err();
        assert_eq!(result.kind(), ErrorKind::IoError);
    }

    #[test]
    #[serial_test::serial]
    fn test_async_cluster_handle_missing_slots_in_split_multi_shard_command() {
        let name = "test_async_cluster_handle_missing_slots_in_split_multi_shard_command";
        let mut cmd = cmd("MGET");
        cmd.arg("foo").arg("bar").arg("baz");
        let MockEnv {
            runtime,
            async_connection: mut connection,
            ..
        } = MockEnv::new(name, move |received_cmd: &[u8], port| {
            respond_startup_with_replica_using_config(
                name,
                received_cmd,
                Some(vec![MockSlotRange {
                    primary_port: 6381,
                    replica_ports: vec![6382],
                    slot_range: (8192..16383),
                }]),
            )?;
            Err(Ok(Value::Array(vec![Value::BulkString(
                format!("{port}").into_bytes(),
            )])))
        });

        let result = runtime
            .block_on(cmd.query_async::<_, Vec<String>>(&mut connection))
            .unwrap_err();
        assert!(
            matches!(result.kind(), ErrorKind::ConnectionNotFoundForRoute)
                || result.is_connection_dropped()
        );
    }

    #[test]
    #[serial_test::serial]
    fn test_async_cluster_with_username_and_password() {
        let cluster = TestClusterContext::new_with_cluster_client_builder(
            3,
            0,
            |builder| {
                builder
                    .username(RedisCluster::username().to_string())
                    .password(RedisCluster::password().to_string())
            },
            false,
        );
        cluster.disable_default_user();

        block_on_all(async move {
            let mut connection = cluster.async_connection(None).await;
            cmd("SET")
                .arg("test")
                .arg("test_data")
                .query_async(&mut connection)
                .await?;
            let res: String = cmd("GET")
                .arg("test")
                .clone()
                .query_async(&mut connection)
                .await?;
            assert_eq!(res, "test_data");
            Ok::<_, RedisError>(())
        })
        .unwrap();
    }

    #[test]
    #[serial_test::serial]
    fn test_async_cluster_io_error() {
        let name = "node";
        let completed = Arc::new(AtomicI32::new(0));
        let MockEnv {
            runtime,
            async_connection: mut connection,
            handler: _handler,
            ..
        } = MockEnv::with_client_builder(
            ClusterClient::builder(vec![&*format!("redis://{name}")]).retries(2),
            name,
            move |cmd: &[u8], port| {
                respond_startup_two_nodes(name, cmd)?;
                // Error twice with io-error, ensure connection is reestablished w/out calling
                // other node (i.e., not doing a full slot rebuild)
                match port {
                    6380 => panic!("Node should not be called"),
                    _ => match completed.fetch_add(1, Ordering::SeqCst) {
                        0..=1 => Err(Err(RedisError::from((
                            ErrorKind::FatalSendError,
                            "mock-io-error",
                        )))),
                        _ => Err(Ok(Value::BulkString(b"123".to_vec()))),
                    },
                }
            },
        );

        let value = runtime.block_on(
            cmd("GET")
                .arg("test")
                .query_async::<_, Option<i32>>(&mut connection),
        );

        assert_eq!(value, Ok(Some(123)));
    }

    #[test]
    #[serial_test::serial]
    fn test_async_cluster_non_retryable_error_should_not_retry() {
        let name = "node";
        let completed = Arc::new(AtomicI32::new(0));
        let MockEnv {
            async_connection: mut connection,
            handler: _handler,
            runtime,
            ..
        } = MockEnv::new(name, {
            let completed = completed.clone();
            move |cmd: &[u8], _| {
                respond_startup_two_nodes(name, cmd)?;
                // Error twice with io-error, ensure connection is reestablished w/out calling
                // other node (i.e., not doing a full slot rebuild)
                completed.fetch_add(1, Ordering::SeqCst);
                Err(Err((ErrorKind::ReadOnly, "").into()))
            }
        });

        let value = runtime.block_on(
            cmd("GET")
                .arg("test")
                .query_async::<_, Option<i32>>(&mut connection),
        );

        match value {
            Ok(_) => panic!("result should be an error"),
            Err(e) => match e.kind() {
                ErrorKind::ReadOnly => {}
                _ => panic!("Expected ReadOnly but got {:?}", e.kind()),
            },
        }
        assert_eq!(completed.load(Ordering::SeqCst), 1);
    }

    #[test]
    #[serial_test::serial]
    fn test_async_cluster_non_retryable_io_error_should_not_retry() {
        let name = "test_async_cluster_non_retryable_io_error_should_not_retry";
        let requests = atomic::AtomicUsize::new(0);
        let MockEnv {
            runtime,
            async_connection: mut connection,
            ..
        } = MockEnv::with_client_builder(
            ClusterClient::builder(vec![&*format!("redis://{name}")]).retries(3),
            name,
            move |cmd: &[u8], _port| {
                respond_startup_two_nodes(name, cmd)?;
                let i = requests.fetch_add(1, atomic::Ordering::SeqCst);
                match i {
                    0 => Err(Err(RedisError::from((ErrorKind::IoError, "io-error")))),
                    _ => {
                        panic!("Expected not to be retried!")
                    }
                }
            },
        );
        runtime
            .block_on(async move {
                let res = cmd("INCR")
                    .arg("foo")
                    .query_async::<_, Option<i32>>(&mut connection)
                    .await;
                assert!(res.is_err());
                let err = res.unwrap_err();
                assert!(err.is_io_error());
                Ok::<_, RedisError>(())
            })
            .unwrap();
    }

    #[test]
    #[serial_test::serial]
    fn test_async_cluster_retry_safe_io_error_should_be_retried() {
        let name = "test_async_cluster_retry_safe_io_error_should_be_retried";
        let requests = atomic::AtomicUsize::new(0);
        let MockEnv {
            runtime,
            async_connection: mut connection,
            ..
        } = MockEnv::with_client_builder(
            ClusterClient::builder(vec![&*format!("redis://{name}")]).retries(3),
            name,
            move |cmd: &[u8], _port| {
                respond_startup_two_nodes(name, cmd)?;
                let i = requests.fetch_add(1, atomic::Ordering::SeqCst);
                match i {
                    0 => Err(Err(RedisError::from((
                        ErrorKind::FatalSendError,
                        "server didn't receive the request, safe to retry",
                    )))),
                    _ => Err(Ok(Value::Int(1))),
                }
            },
        );
        runtime
            .block_on(async move {
                let res = cmd("INCR")
                    .arg("foo")
                    .query_async::<_, i32>(&mut connection)
                    .await;
                assert!(res.is_ok());
                let value = res.unwrap();
                assert_eq!(value, 1);
                Ok::<_, RedisError>(())
            })
            .unwrap();
    }

    #[test]
    #[serial_test::serial]
    fn test_async_cluster_read_from_primary() {
        let name = "node";
        let found_ports = Arc::new(std::sync::Mutex::new(Vec::new()));
        let ports_clone = found_ports.clone();
        let MockEnv {
            runtime,
            async_connection: mut connection,
            ..
        } = MockEnv::new(name, move |received_cmd: &[u8], port| {
            respond_startup_with_replica_using_config(
                name,
                received_cmd,
                Some(vec![
                    MockSlotRange {
                        primary_port: 6379,
                        replica_ports: vec![6380, 6381],
                        slot_range: (0..8191),
                    },
                    MockSlotRange {
                        primary_port: 6382,
                        replica_ports: vec![6383, 6384],
                        slot_range: (8192..16383),
                    },
                ]),
            )?;
            ports_clone.lock().unwrap().push(port);
            Err(Ok(Value::Nil))
        });

        runtime.block_on(async {
            cmd("GET")
                .arg("foo")
                .query_async::<_, ()>(&mut connection)
                .await
                .unwrap();
            cmd("GET")
                .arg("bar")
                .query_async::<_, ()>(&mut connection)
                .await
                .unwrap();
            cmd("GET")
                .arg("foo")
                .query_async::<_, ()>(&mut connection)
                .await
                .unwrap();
            cmd("GET")
                .arg("bar")
                .query_async::<_, ()>(&mut connection)
                .await
                .unwrap();
        });

        found_ports.lock().unwrap().sort();
        assert_eq!(*found_ports.lock().unwrap(), vec![6379, 6379, 6382, 6382]);
    }

    #[test]
    #[serial_test::serial]
    fn test_async_cluster_round_robin_read_from_replica() {
        let name = "node";
        let found_ports = Arc::new(std::sync::Mutex::new(Vec::new()));
        let ports_clone = found_ports.clone();
        let MockEnv {
            runtime,
            async_connection: mut connection,
            ..
        } = MockEnv::with_client_builder(
            ClusterClient::builder(vec![&*format!("redis://{name}")]).read_from_replicas(),
            name,
            move |received_cmd: &[u8], port| {
                respond_startup_with_replica_using_config(
                    name,
                    received_cmd,
                    Some(vec![
                        MockSlotRange {
                            primary_port: 6379,
                            replica_ports: vec![6380, 6381],
                            slot_range: (0..8191),
                        },
                        MockSlotRange {
                            primary_port: 6382,
                            replica_ports: vec![6383, 6384],
                            slot_range: (8192..16383),
                        },
                    ]),
                )?;
                ports_clone.lock().unwrap().push(port);
                Err(Ok(Value::Nil))
            },
        );

        runtime.block_on(async {
            cmd("GET")
                .arg("foo")
                .query_async::<_, ()>(&mut connection)
                .await
                .unwrap();
            cmd("GET")
                .arg("bar")
                .query_async::<_, ()>(&mut connection)
                .await
                .unwrap();
            cmd("GET")
                .arg("foo")
                .query_async::<_, ()>(&mut connection)
                .await
                .unwrap();
            cmd("GET")
                .arg("bar")
                .query_async::<_, ()>(&mut connection)
                .await
                .unwrap();
        });

        found_ports.lock().unwrap().sort();
        assert_eq!(*found_ports.lock().unwrap(), vec![6380, 6381, 6383, 6384]);
    }

    fn get_queried_node_id_if_master(cluster_nodes_output: Value) -> Option<String> {
        // Returns the node ID of the connection that was queried for CLUSTER NODES (using the 'myself' flag), if it's a master.
        // Otherwise, returns None.
        let get_node_id = |str: &str| {
            let parts: Vec<&str> = str.split('\n').collect();
            for node_entry in parts {
                if node_entry.contains("myself") && node_entry.contains("master") {
                    let node_entry_parts: Vec<&str> = node_entry.split(' ').collect();
                    let node_id = node_entry_parts[0];
                    return Some(node_id.to_string());
                }
            }
            None
        };

        match cluster_nodes_output {
            Value::BulkString(val) => match from_utf8(&val) {
                Ok(str_res) => get_node_id(str_res),
                Err(e) => panic!("failed to decode INFO response: {:?}", e),
            },
            Value::VerbatimString { format: _, text } => get_node_id(&text),
            _ => panic!("Recieved unexpected response: {:?}", cluster_nodes_output),
        }
    }

    #[test]
    #[serial_test::serial]
    fn test_async_cluster_handle_complete_server_disconnect_without_panicking() {
        let cluster = TestClusterContext::new_with_cluster_client_builder(
            3,
            0,
            |builder| builder.retries(2),
            false,
        );
        block_on_all(async move {
            let mut connection = cluster.async_connection(None).await;
            drop(cluster);
            for _ in 0..5 {
                let cmd = cmd("PING");
                let result = connection
                    .route_command(&cmd, RoutingInfo::SingleNode(SingleNodeRoutingInfo::Random))
                    .await;
                // TODO - this should be a NoConnectionError, but ATM we get the errors from the failing
                assert!(result.is_err());
                // This will route to all nodes - different path through the code.
                let result = connection.req_packed_command(&cmd).await;
                // TODO - this should be a NoConnectionError, but ATM we get the errors from the failing
                assert!(result.is_err());
            }
            Ok::<_, RedisError>(())
        })
        .unwrap();
    }

    #[test]
    #[serial_test::serial]
    fn test_async_cluster_test_fast_reconnect() {
        // Note the 3 seconds connection check to differentiate between notifications and periodic
        let cluster = TestClusterContext::new_with_cluster_client_builder(
            3,
            0,
            |builder| {
                builder
                    .retries(0)
                    .periodic_connections_checks(Duration::from_secs(3))
            },
            false,
        );

        // For tokio-comp, do 3 consequtive disconnects and ensure reconnects succeeds in less than 100ms,
        // which is more than enough for local connections even with TLS.
        // More than 1 run is done to ensure it is the fast reconnect notification that trigger the reconnect
        // and not the periodic interval.
        // For other async implementation, only periodic connection check is available, hence,
        // do 1 run sleeping for periodic connection check interval, allowing it to reestablish connections
        block_on_all(async move {
            let mut disconnecting_con = cluster.async_connection(None).await;
            let mut monitoring_con = cluster.async_connection(None).await;

            #[cfg(feature = "tokio-comp")]
            let tries = 0..3;
            #[cfg(not(feature = "tokio-comp"))]
            let tries = 0..1;

            for _ in tries {
                // get connection id
                let mut cmd = redis::cmd("CLIENT");
                cmd.arg("ID");
                let res = disconnecting_con
                    .route_command(
                        &cmd,
                        RoutingInfo::SingleNode(SingleNodeRoutingInfo::SpecificNode(Route::new(
                            0,
                            SlotAddr::Master,
                        ))),
                    )
                    .await;
                assert!(res.is_ok());
                let res = res.unwrap();
                let id = {
                    match res {
                        Value::Int(id) => id,
                        _ => {
                            panic!("Wrong return value for CLIENT ID command: {:?}", res);
                        }
                    }
                };

                // ask server to kill the connection
                let mut cmd = redis::cmd("CLIENT");
                cmd.arg("KILL").arg("ID").arg(id).arg("SKIPME").arg("NO");
                let res = disconnecting_con
                    .route_command(
                        &cmd,
                        RoutingInfo::SingleNode(SingleNodeRoutingInfo::SpecificNode(Route::new(
                            0,
                            SlotAddr::Master,
                        ))),
                    )
                    .await;
                // assert server has closed connection
                assert_eq!(res, Ok(Value::Int(1)));

                #[cfg(feature = "tokio-comp")]
                // ensure reconnect happened in less than 100ms
                sleep(futures_time::time::Duration::from_millis(100)).await;

                #[cfg(not(feature = "tokio-comp"))]
                // no fast notification is available, wait for 1 periodic check + overhead
                sleep(futures_time::time::Duration::from_secs(3 + 1)).await;

                let mut cmd = redis::cmd("CLIENT");
                cmd.arg("LIST").arg("TYPE").arg("NORMAL");
                let res = monitoring_con
                    .route_command(
                        &cmd,
                        RoutingInfo::SingleNode(SingleNodeRoutingInfo::SpecificNode(Route::new(
                            0,
                            SlotAddr::Master,
                        ))),
                    )
                    .await;
                assert!(res.is_ok());
                let res = res.unwrap();
                let client_list: String = {
                    match res {
                        // RESP2
                        Value::BulkString(client_info) => {
                            // ensure 4 connections - 2 for each client, its save to unwrap here
                            String::from_utf8(client_info).unwrap()
                        }
                        // RESP3
                        Value::VerbatimString { format: _, text } => text,
                        _ => {
                            panic!("Wrong return type for CLIENT LIST command: {:?}", res);
                        }
                    }
                };
                assert_eq!(client_list.chars().filter(|&x| x == '\n').count(), 4);
            }
            Ok(())
        })
        .unwrap();
    }

    #[test]
    #[serial_test::serial]
    fn test_async_cluster_restore_resp3_pubsub_state_passive_disconnect() {
        let redis_ver = std::env::var("REDIS_VERSION").unwrap_or_default();
        let use_sharded = redis_ver.starts_with("7.");

        let mut client_subscriptions = PubSubSubscriptionInfo::from([(
            PubSubSubscriptionKind::Exact,
            HashSet::from([PubSubChannelOrPattern::from("test_channel".as_bytes())]),
        )]);

        if use_sharded {
            client_subscriptions.insert(
                PubSubSubscriptionKind::Sharded,
                HashSet::from([PubSubChannelOrPattern::from("test_channel_?".as_bytes())]),
            );
        }

        // note topology change detection is not activated since no topology change is expected
        let cluster = TestClusterContext::new_with_cluster_client_builder(
            3,
            0,
            |builder| {
                builder
                    .retries(3)
                    .use_protocol(ProtocolVersion::RESP3)
                    .pubsub_subscriptions(client_subscriptions.clone())
                    .periodic_connections_checks(Duration::from_secs(1))
            },
            false,
        );

        block_on_all(async move {
            let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<PushInfo>();
            let mut _listening_con = cluster.async_connection(Some(tx.clone())).await;
            // Note, publishing connection has the same pubsub config
            let mut publishing_con = cluster.async_connection(None).await;

            // short sleep to allow the server to push subscription notification
            sleep(futures_time::time::Duration::from_secs(1)).await;

            // validate subscriptions
            validate_subscriptions(&client_subscriptions, &mut rx, false);

            // validate PUBLISH
            let result = cmd("PUBLISH")
                .arg("test_channel")
                .arg("test_message")
                .query_async(&mut publishing_con)
                .await;
            assert_eq!(
                result,
                Ok(Value::Int(2)) // 2 connections with the same pubsub config
            );

            sleep(futures_time::time::Duration::from_secs(1)).await;
            let result = rx.try_recv();
            assert!(result.is_ok());
            let PushInfo { kind, data } = result.unwrap();
            assert_eq!(
                (kind, data),
                (
                    PushKind::Message,
                    vec![
                        Value::BulkString("test_channel".into()),
                        Value::BulkString("test_message".into()),
                    ]
                )
            );

            if use_sharded {
                // validate SPUBLISH
                let result = cmd("SPUBLISH")
                    .arg("test_channel_?")
                    .arg("test_message")
                    .query_async(&mut publishing_con)
                    .await;
                assert_eq!(
                    result,
                    Ok(Value::Int(2)) // 2 connections with the same pubsub config
                );

                sleep(futures_time::time::Duration::from_secs(1)).await;
                let result = rx.try_recv();
                assert!(result.is_ok());
                let PushInfo { kind, data } = result.unwrap();
                assert_eq!(
                    (kind, data),
                    (
                        PushKind::SMessage,
                        vec![
                            Value::BulkString("test_channel_?".into()),
                            Value::BulkString("test_message".into()),
                        ]
                    )
                );
            }

            // simulate passive disconnect
            drop(cluster);

            // recreate the cluster, the assumtion is that the cluster is built with exactly the same params (ports, slots map...)
            let _cluster =
                TestClusterContext::new_with_cluster_client_builder(3, 0, |builder| builder, false);

            // sleep for 1 periodic_connections_checks + overhead
            sleep(futures_time::time::Duration::from_secs(1 + 1)).await;

            // new subscription notifications due to resubscriptions
            validate_subscriptions(&client_subscriptions, &mut rx, true);

            // validate PUBLISH
            let result = cmd("PUBLISH")
                .arg("test_channel")
                .arg("test_message")
                .query_async(&mut publishing_con)
                .await;
            assert_eq!(
                result,
                Ok(Value::Int(2)) // 2 connections with the same pubsub config
            );

            sleep(futures_time::time::Duration::from_secs(1)).await;
            let result = rx.try_recv();
            assert!(result.is_ok());
            let PushInfo { kind, data } = result.unwrap();
            assert_eq!(
                (kind, data),
                (
                    PushKind::Message,
                    vec![
                        Value::BulkString("test_channel".into()),
                        Value::BulkString("test_message".into()),
                    ]
                )
            );

            if use_sharded {
                // validate SPUBLISH
                let result = cmd("SPUBLISH")
                    .arg("test_channel_?")
                    .arg("test_message")
                    .query_async(&mut publishing_con)
                    .await;
                assert_eq!(
                    result,
                    Ok(Value::Int(2)) // 2 connections with the same pubsub config
                );

                sleep(futures_time::time::Duration::from_secs(1)).await;
                let result = rx.try_recv();
                assert!(result.is_ok());
                let PushInfo { kind, data } = result.unwrap();
                assert_eq!(
                    (kind, data),
                    (
                        PushKind::SMessage,
                        vec![
                            Value::BulkString("test_channel_?".into()),
                            Value::BulkString("test_message".into()),
                        ]
                    )
                );
            }

            Ok(())
        })
        .unwrap();
    }

    #[test]
    #[serial_test::serial]
    fn test_async_cluster_restore_resp3_pubsub_state_after_scale_out() {
        let redis_ver = std::env::var("REDIS_VERSION").unwrap_or_default();
        let use_sharded = redis_ver.starts_with("7.");

        let mut client_subscriptions = PubSubSubscriptionInfo::from([
            // test_channel_? is used as it maps to 14212 slot, which is the last node in both 3 and 6 node config
            // (assuming slots allocation is monotonicaly increasing starting from node 0)
            (
                PubSubSubscriptionKind::Exact,
                HashSet::from([PubSubChannelOrPattern::from("test_channel_?".as_bytes())]),
            ),
        ]);

        if use_sharded {
            client_subscriptions.insert(
                PubSubSubscriptionKind::Sharded,
                HashSet::from([PubSubChannelOrPattern::from("test_channel_?".as_bytes())]),
            );
        }

        let slot_14212 = get_slot(b"test_channel_?");
        assert_eq!(slot_14212, 14212);

        let cluster = TestClusterContext::new_with_cluster_client_builder(
            3,
            0,
            |builder| {
                builder
            .retries(3)
            .use_protocol(ProtocolVersion::RESP3)
            .pubsub_subscriptions(client_subscriptions.clone())
            // periodic connection check is required to detect the disconnect from the last node
            .periodic_connections_checks(Duration::from_secs(1))
            // periodic topology check is required to detect topology change
            .periodic_topology_checks(Duration::from_secs(1))
            .slots_refresh_rate_limit(Duration::from_secs(0), 0)
            },
            false,
        );

        block_on_all(async move {
            let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<PushInfo>();
            let mut _listening_con = cluster.async_connection(Some(tx.clone())).await;
            // Note, publishing connection has the same pubsub config
            let mut publishing_con = cluster.async_connection(None).await;

            // short sleep to allow the server to push subscription notification
            sleep(futures_time::time::Duration::from_secs(1)).await;

            // validate subscriptions
            validate_subscriptions(&client_subscriptions, &mut rx, false);

            // validate PUBLISH
            let result = cmd("PUBLISH")
                .arg("test_channel_?")
                .arg("test_message")
                .query_async(&mut publishing_con)
                .await;
            assert_eq!(
                result,
                Ok(Value::Int(2)) // 2 connections with the same pubsub config
            );

            sleep(futures_time::time::Duration::from_secs(1)).await;
            let result = rx.try_recv();
            assert!(result.is_ok());
            let PushInfo { kind, data } = result.unwrap();
            assert_eq!(
                (kind, data),
                (
                    PushKind::Message,
                    vec![
                        Value::BulkString("test_channel_?".into()),
                        Value::BulkString("test_message".into()),
                    ]
                )
            );

            if use_sharded {
                // validate SPUBLISH
                let result = cmd("SPUBLISH")
                    .arg("test_channel_?")
                    .arg("test_message")
                    .query_async(&mut publishing_con)
                    .await;
                assert_eq!(
                    result,
                    Ok(Value::Int(2)) // 2 connections with the same pubsub config
                );

                sleep(futures_time::time::Duration::from_secs(1)).await;
                let result = rx.try_recv();
                assert!(result.is_ok());
                let PushInfo { kind, data } = result.unwrap();
                assert_eq!(
                    (kind, data),
                    (
                        PushKind::SMessage,
                        vec![
                            Value::BulkString("test_channel_?".into()),
                            Value::BulkString("test_message".into()),
                        ]
                    )
                );
            }

            // drop and recreate a cluster with more nodes
            drop(cluster);

            // recreate the cluster, the assumtion is that the cluster is built with exactly the same params (ports, slots map...)
            let cluster =
                TestClusterContext::new_with_cluster_client_builder(6, 0, |builder| builder, false);

            // assume slot 14212 will reside in the last node
            let last_server_port = {
                let addr = cluster.cluster.servers.last().unwrap().addr.clone();
                match addr {
                    redis::ConnectionAddr::TcpTls {
                        host: _,
                        port,
                        insecure: _,
                        tls_params: _,
                    } => port,
                    redis::ConnectionAddr::Tcp(_, port) => port,
                    _ => {
                        panic!("Wrong server address type: {:?}", addr);
                    }
                }
            };

            // wait for new topology discovery
            let max_requests = 5;
            let mut i = 0;
            let mut cmd = redis::cmd("INFO");
            cmd.arg("SERVER");
            loop {
                if i == max_requests {
                    panic!("Failed to recover and discover new topology");
                }
                i += 1;

                if let Ok(res) = publishing_con
                    .route_command(
                        &cmd,
                        RoutingInfo::SingleNode(SingleNodeRoutingInfo::SpecificNode(Route::new(
                            slot_14212,
                            SlotAddr::Master,
                        ))),
                    )
                    .await
                {
                    match res {
                        Value::VerbatimString { format: _, text } => {
                            if text.contains(format!("tcp_port:{}", last_server_port).as_str()) {
                                // new topology rediscovered
                                break;
                            }
                        }
                        _ => {
                            panic!("Wrong return type for INFO SERVER command: {:?}", res);
                        }
                    }
                    sleep(futures_time::time::Duration::from_secs(1)).await;
                }
            }

            // sleep for one one cycle of topology refresh
            sleep(futures_time::time::Duration::from_secs(1)).await;

            // validate PUBLISH
            let result = redis::cmd("PUBLISH")
                .arg("test_channel_?")
                .arg("test_message")
                .query_async(&mut publishing_con)
                .await;
            assert_eq!(
                result,
                Ok(Value::Int(2)) // 2 connections with the same pubsub config
            );

            // allow message to propagate
            sleep(futures_time::time::Duration::from_secs(1)).await;

            loop {
                let result = rx.try_recv();
                assert!(result.is_ok());
                let PushInfo { kind, data } = result.unwrap();
                // ignore disconnection and subscription notifications due to resubscriptions
                if kind == PushKind::Message {
                    assert_eq!(
                        data,
                        vec![
                            Value::BulkString("test_channel_?".into()),
                            Value::BulkString("test_message".into()),
                        ]
                    );
                    break;
                }
            }

            if use_sharded {
                // validate SPUBLISH
                let result = redis::cmd("SPUBLISH")
                    .arg("test_channel_?")
                    .arg("test_message")
                    .query_async(&mut publishing_con)
                    .await;
                assert_eq!(
                    result,
                    Ok(Value::Int(2)) // 2 connections with the same pubsub config
                );

                // allow message to propagate
                sleep(futures_time::time::Duration::from_secs(1)).await;

                let result = rx.try_recv();
                assert!(result.is_ok());
                let PushInfo { kind, data } = result.unwrap();
                assert_eq!(
                    (kind, data),
                    (
                        PushKind::SMessage,
                        vec![
                            Value::BulkString("test_channel_?".into()),
                            Value::BulkString("test_message".into()),
                        ]
                    )
                );
            }

            drop(publishing_con);
            drop(_listening_con);

            Ok(())
        })
        .unwrap();

        block_on_all(async move {
            sleep(futures_time::time::Duration::from_secs(10)).await;
            Ok(())
        })
        .unwrap();
    }

    #[test]
    #[serial_test::serial]
    fn test_async_cluster_resp3_pubsub() {
        let redis_ver = std::env::var("REDIS_VERSION").unwrap_or_default();
        let use_sharded = redis_ver.starts_with("7.");

        let mut client_subscriptions = PubSubSubscriptionInfo::from([
            (
                PubSubSubscriptionKind::Exact,
                HashSet::from([PubSubChannelOrPattern::from("test_channel_?".as_bytes())]),
            ),
            (
                PubSubSubscriptionKind::Pattern,
                HashSet::from([
                    PubSubChannelOrPattern::from("test_*".as_bytes()),
                    PubSubChannelOrPattern::from("*".as_bytes()),
                ]),
            ),
        ]);

        if use_sharded {
            client_subscriptions.insert(
                PubSubSubscriptionKind::Sharded,
                HashSet::from([PubSubChannelOrPattern::from("test_channel_?".as_bytes())]),
            );
        }

        let cluster = TestClusterContext::new_with_cluster_client_builder(
            3,
            0,
            |builder| {
                builder
                    .retries(3)
                    .use_protocol(ProtocolVersion::RESP3)
                    .pubsub_subscriptions(client_subscriptions.clone())
            },
            false,
        );

        block_on_all(async move {
            let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<PushInfo>();
            let mut connection = cluster.async_connection(Some(tx.clone())).await;

            // short sleep to allow the server to push subscription notification
            sleep(futures_time::time::Duration::from_secs(1)).await;

            validate_subscriptions(&client_subscriptions, &mut rx, false);

            let slot_14212 = get_slot(b"test_channel_?");
            assert_eq!(slot_14212, 14212);

            let slot_0_route =
                redis::cluster_routing::Route::new(0, redis::cluster_routing::SlotAddr::Master);
            let node_0_route =
                redis::cluster_routing::SingleNodeRoutingInfo::SpecificNode(slot_0_route);

            // node 0 route is used to ensure that the publish is propagated correctly
            let result = connection
                .route_command(
                    redis::Cmd::new()
                        .arg("PUBLISH")
                        .arg("test_channel_?")
                        .arg("test_message"),
                    RoutingInfo::SingleNode(node_0_route.clone()),
                )
                .await;
            assert!(result.is_ok());

            sleep(futures_time::time::Duration::from_secs(1)).await;

            let mut pmsg_cnt = 0;
            let mut msg_cnt = 0;
            for _ in 0..3 {
                let result = rx.try_recv();
                assert!(result.is_ok());
                let PushInfo { kind, data: _ } = result.unwrap();
                assert!(kind == PushKind::Message || kind == PushKind::PMessage);
                if kind == PushKind::Message {
                    msg_cnt += 1;
                } else {
                    pmsg_cnt += 1;
                }
            }
            assert_eq!(msg_cnt, 1);
            assert_eq!(pmsg_cnt, 2);

            if use_sharded {
                let result = cmd("SPUBLISH")
                    .arg("test_channel_?")
                    .arg("test_message")
                    .query_async(&mut connection)
                    .await;
                assert_eq!(result, Ok(Value::Int(1)));

                sleep(futures_time::time::Duration::from_secs(1)).await;
                let result = rx.try_recv();
                assert!(result.is_ok());
                let PushInfo { kind, data } = result.unwrap();
                assert_eq!(
                    (kind, data),
                    (
                        PushKind::SMessage,
                        vec![
                            Value::BulkString("test_channel_?".into()),
                            Value::BulkString("test_message".into()),
                        ]
                    )
                );
            }

            Ok(())
        })
        .unwrap();
    }

    #[test]
    #[serial_test::serial]
    fn test_async_cluster_periodic_checks_update_topology_after_failover() {
        // This test aims to validate the functionality of periodic topology checks by detecting and updating topology changes.
        // We will repeatedly execute CLUSTER NODES commands against the primary node responsible for slot 0, recording its node ID.
        // Once we've successfully completed commands with the current primary, we will initiate a failover within the same shard.
        // Since we are not executing key-based commands, we won't encounter MOVED errors that trigger a slot refresh.
        // Consequently, we anticipate that only the periodic topology check will detect this change and trigger topology refresh.
        // If successful, the node to which we route the CLUSTER NODES command should be the newly promoted node with a different node ID.
        let cluster = TestClusterContext::new_with_cluster_client_builder(
            6,
            1,
            |builder| {
                builder
                    .periodic_topology_checks(Duration::from_millis(10))
                    // Disable the rate limiter to refresh slots immediately on all MOVED errors
                    .slots_refresh_rate_limit(Duration::from_secs(0), 0)
            },
            false,
        );

        block_on_all(async move {
            let mut connection = cluster.async_connection(None).await;
            let mut prev_master_id = "".to_string();
            let max_requests = 5000;
            let mut i = 0;
            loop {
                if i == 10 {
                    let mut cmd = redis::cmd("CLUSTER");
                    cmd.arg("FAILOVER");
                    cmd.arg("TAKEOVER");
                    let res = connection
                        .route_command(
                            &cmd,
                            RoutingInfo::SingleNode(SingleNodeRoutingInfo::SpecificNode(
                                Route::new(0, SlotAddr::ReplicaRequired),
                            )),
                        )
                        .await;
                    assert!(res.is_ok());
                } else if i == max_requests {
                    break;
                } else {
                    let mut cmd = redis::cmd("CLUSTER");
                    cmd.arg("NODES");
                    let res = connection
                        .route_command(
                            &cmd,
                            RoutingInfo::SingleNode(SingleNodeRoutingInfo::SpecificNode(
                                Route::new(0, SlotAddr::Master),
                            )),
                        )
                        .await
                        .expect("Failed executing CLUSTER NODES");
                    let node_id = get_queried_node_id_if_master(res);
                    if let Some(current_master_id) = node_id {
                        if prev_master_id.is_empty() {
                            prev_master_id = current_master_id;
                        } else if prev_master_id != current_master_id {
                            return Ok::<_, RedisError>(());
                        }
                    }
                }
                i += 1;
                let _ = sleep(futures_time::time::Duration::from_millis(10)).await;
            }
            panic!("Topology change wasn't found!");
        })
        .unwrap();
    }

    #[test]
    #[serial_test::serial]
    fn test_async_cluster_recover_disconnected_management_connections() {
        // This test aims to verify that the management connections used for periodic checks are reconnected, in case that they get killed.
        // In order to test this, we choose a single node, kill all connections to it which aren't user connections, and then wait until new
        // connections are created.
        let cluster = TestClusterContext::new_with_cluster_client_builder(
            3,
            0,
            |builder| {
                builder.periodic_topology_checks(Duration::from_millis(10))
                            // Disable the rate limiter to refresh slots immediately
                            .slots_refresh_rate_limit(Duration::from_secs(0), 0)
            },
            false,
        );

        block_on_all(async move {
            let routing = RoutingInfo::SingleNode(SingleNodeRoutingInfo::SpecificNode(Route::new(
                1,
                SlotAddr::Master,
            )));

            let mut connection = cluster.async_connection(None).await;
            let max_requests = 5000;

            let connections =
                get_clients_names_to_ids(&mut connection, routing.clone().into()).await;
            assert!(connections.contains_key(MANAGEMENT_CONN_NAME));
            let management_conn_id = connections.get(MANAGEMENT_CONN_NAME).unwrap();

            // Get the connection ID of the management connection
            kill_connection(&mut connection, management_conn_id).await;

            let connections =
                get_clients_names_to_ids(&mut connection, routing.clone().into()).await;
            assert!(!connections.contains_key(MANAGEMENT_CONN_NAME));

            for _ in 0..max_requests {
                let _ = sleep(futures_time::time::Duration::from_millis(10)).await;

                let connections =
                    get_clients_names_to_ids(&mut connection, routing.clone().into()).await;
                if connections.contains_key(MANAGEMENT_CONN_NAME) {
                    return Ok(());
                }
            }

            panic!("Topology connection didn't reconnect!");
        })
        .unwrap();
    }

    #[test]
    #[serial_test::serial]
    fn test_async_cluster_with_client_name() {
        let cluster = TestClusterContext::new_with_cluster_client_builder(
            3,
            0,
            |builder| builder.client_name(RedisCluster::client_name().to_string()),
            false,
        );

        block_on_all(async move {
            let mut connection = cluster.async_connection(None).await;
            let client_info: String = cmd("CLIENT")
                .arg("INFO")
                .query_async(&mut connection)
                .await
                .unwrap();

            let client_attrs = parse_client_info(&client_info);

            assert!(
                client_attrs.contains_key("name"),
                "Could not detect the 'name' attribute in CLIENT INFO output"
            );

            assert_eq!(
                client_attrs["name"],
                RedisCluster::client_name(),
                "Incorrect client name, expecting: {}, got {}",
                RedisCluster::client_name(),
                client_attrs["name"]
            );
            Ok::<_, RedisError>(())
        })
        .unwrap();
    }

    #[test]
    #[serial_test::serial]
    fn test_async_cluster_reroute_from_replica_if_in_loading_state() {
        /* Test replica in loading state. The expected behaviour is that the request will be directed to a different replica or the primary.
        depends on the read from replica policy. */
        let name = "test_async_cluster_reroute_from_replica_if_in_loading_state";

        let load_errors: Arc<_> = Arc::new(std::sync::Mutex::new(vec![]));
        let load_errors_clone = load_errors.clone();

        // requests should route to replica
        let MockEnv {
            runtime,
            async_connection: mut connection,
            handler: _handler,
            ..
        } = MockEnv::with_client_builder(
            ClusterClient::builder(vec![&*format!("redis://{name}")]).read_from_replicas(),
            name,
            move |cmd: &[u8], port| {
                respond_startup_with_replica_using_config(
                    name,
                    cmd,
                    Some(vec![MockSlotRange {
                        primary_port: 6379,
                        replica_ports: vec![6380, 6381],
                        slot_range: (0..16383),
                    }]),
                )?;
                match port {
                    6380 | 6381 => {
                        load_errors_clone.lock().unwrap().push(port);
                        Err(parse_redis_value(b"-LOADING\r\n"))
                    }
                    6379 => Err(Ok(Value::BulkString(b"123".to_vec()))),
                    _ => panic!("Wrong node"),
                }
            },
        );
        for _n in 0..3 {
            let value = runtime.block_on(
                cmd("GET")
                    .arg("test")
                    .query_async::<_, Option<i32>>(&mut connection),
            );
            assert_eq!(value, Ok(Some(123)));
        }

        let mut load_errors_guard = load_errors.lock().unwrap();
        load_errors_guard.sort();

        // We expected to get only 2 loading error since the 2 replicas are in loading state.
        // The third iteration will be directed to the primary since the connections of the replicas were removed.
        assert_eq!(*load_errors_guard, vec![6380, 6381]);
    }

    #[test]
    #[serial_test::serial]
    fn test_async_cluster_read_from_primary_when_primary_loading() {
        // Test primary in loading state. The expected behaviour is that the request will be retried until the primary is no longer in loading state.
        let name = "test_async_cluster_read_from_primary_when_primary_loading";

        const RETRIES: u32 = 3;
        const ITERATIONS: u32 = 2;
        let load_errors = Arc::new(AtomicU32::new(0));
        let load_errors_clone = load_errors.clone();

        let MockEnv {
            runtime,
            async_connection: mut connection,
            handler: _handler,
            ..
        } = MockEnv::with_client_builder(
            ClusterClient::builder(vec![&*format!("redis://{name}")]),
            name,
            move |cmd: &[u8], port| {
                respond_startup_with_replica_using_config(
                    name,
                    cmd,
                    Some(vec![MockSlotRange {
                        primary_port: 6379,
                        replica_ports: vec![6380, 6381],
                        slot_range: (0..16383),
                    }]),
                )?;
                match port {
                    6379 => {
                        let attempts = load_errors_clone.fetch_add(1, Ordering::Relaxed) + 1;
                        if attempts % RETRIES == 0 {
                            Err(Ok(Value::BulkString(b"123".to_vec())))
                        } else {
                            Err(parse_redis_value(b"-LOADING\r\n"))
                        }
                    }
                    _ => panic!("Wrong node"),
                }
            },
        );
        for _n in 0..ITERATIONS {
            runtime
                .block_on(
                    cmd("GET")
                        .arg("test")
                        .query_async::<_, Value>(&mut connection),
                )
                .unwrap();
        }

        assert_eq!(load_errors.load(Ordering::Relaxed), ITERATIONS * RETRIES);
    }

    #[test]
    #[serial_test::serial]
    fn test_async_cluster_can_be_created_with_partial_slot_coverage() {
        let name = "test_async_cluster_can_be_created_with_partial_slot_coverage";
        let slots_config = Some(vec![
            MockSlotRange {
                primary_port: 6379,
                replica_ports: vec![],
                slot_range: (0..8000),
            },
            MockSlotRange {
                primary_port: 6381,
                replica_ports: vec![],
                slot_range: (8201..16380),
            },
        ]);

        let MockEnv {
            async_connection: mut connection,
            handler: _handler,
            runtime,
            ..
        } = MockEnv::with_client_builder(
            ClusterClient::builder(vec![&*format!("redis://{name}")])
                .retries(0)
                .read_from_replicas(),
            name,
            move |received_cmd: &[u8], _| {
                respond_startup_with_replica_using_config(
                    name,
                    received_cmd,
                    slots_config.clone(),
                )?;
                Err(Ok(Value::SimpleString("PONG".into())))
            },
        );

        let res = runtime.block_on(connection.req_packed_command(&redis::cmd("PING")));
        assert!(res.is_ok());
    }

    #[test]
    #[serial_test::serial]
    fn test_async_cluster_reconnect_after_complete_server_disconnect() {
        let cluster = TestClusterContext::new_with_cluster_client_builder(
            3,
            0,
            |builder| {
                builder.retries(2)
                // Disable the rate limiter to refresh slots immediately
                .slots_refresh_rate_limit(Duration::from_secs(0), 0)
            },
            false,
        );

        block_on_all(async move {
            let mut connection = cluster.async_connection(None).await;
            drop(cluster);
            let cmd = cmd("PING");

            let result = connection
                .route_command(&cmd, RoutingInfo::SingleNode(SingleNodeRoutingInfo::Random))
                .await;
            // TODO - this should be a NoConnectionError, but ATM we get the errors from the failing
            assert!(result.is_err());

            // This will route to all nodes - different path through the code.
            let result = connection.req_packed_command(&cmd).await;
            // TODO - this should be a NoConnectionError, but ATM we get the errors from the failing
            assert!(result.is_err());

            let _cluster = TestClusterContext::new_with_cluster_client_builder(
                3,
                0,
                |builder| builder.retries(2),
                false,
            );

            let max_requests = 5;
            let mut i = 0;
            let mut last_err = None;
            loop {
                if i == max_requests {
                    break;
                }
                i += 1;
                match connection.req_packed_command(&cmd).await {
                    Ok(result) => {
                        assert_eq!(result, Value::SimpleString("PONG".to_string()));
                        return Ok::<_, RedisError>(());
                    }
                    Err(err) => {
                        last_err = Some(err);
                        let _ = sleep(futures_time::time::Duration::from_secs(1)).await;
                    }
                }
            }
            panic!("Failed to recover after all nodes went down. Last error: {last_err:?}");
        })
        .unwrap();
    }

    #[test]
    #[serial_test::serial]
    fn test_async_cluster_reconnect_after_complete_server_disconnect_route_to_many() {
        let cluster = TestClusterContext::new_with_cluster_client_builder(
            3,
            0,
            |builder| builder.retries(3),
            false,
        );
        block_on_all(async move {
            let mut connection = cluster.async_connection(None).await;
            drop(cluster);

            // recreate cluster
            let _cluster = TestClusterContext::new_with_cluster_client_builder(
                3,
                0,
                |builder| builder.retries(2),
                false,
            );

            let cmd = cmd("PING");

            let max_requests = 5;
            let mut i = 0;
            let mut last_err = None;
            loop {
                if i == max_requests {
                    break;
                }
                i += 1;
                // explicitly route to all primaries and request all succeeded
                match connection
                    .route_command(
                        &cmd,
                        RoutingInfo::MultiNode((
                            MultipleNodeRoutingInfo::AllMasters,
                            Some(redis::cluster_routing::ResponsePolicy::AllSucceeded),
                        )),
                    )
                    .await
                {
                    Ok(result) => {
                        assert_eq!(result, Value::SimpleString("PONG".to_string()));
                        return Ok::<_, RedisError>(());
                    }
                    Err(err) => {
                        last_err = Some(err);
                        let _ = sleep(futures_time::time::Duration::from_secs(1)).await;
                    }
                }
            }
            panic!("Failed to recover after all nodes went down. Last error: {last_err:?}");
        })
        .unwrap();
    }

    #[test]
    #[serial_test::serial]
    fn test_async_cluster_blocking_command_when_cluster_drops() {
        let cluster = TestClusterContext::new_with_cluster_client_builder(
            3,
            0,
            |builder| builder.retries(3),
            false,
        );
        block_on_all(async move {
            let mut connection = cluster.async_connection(None).await;
            futures::future::join(
                async {
                    let res = connection.blpop::<&str, f64>("foo", 0.0).await;
                    assert!(res.is_err());
                    println!("blpop returned error {:?}", res.map_err(|e| e.to_string()));
                },
                async {
                    let _ = sleep(futures_time::time::Duration::from_secs(3)).await;
                    drop(cluster);
                },
            )
            .await;
            Ok::<_, RedisError>(())
        })
        .unwrap();
    }

    #[test]
    #[serial_test::serial]
    fn test_async_cluster_saves_reconnected_connection() {
        let name = "test_async_cluster_saves_reconnected_connection";
        let ping_attempts = Arc::new(AtomicI32::new(0));
        let ping_attempts_clone = ping_attempts.clone();
        let get_attempts = AtomicI32::new(0);

        let MockEnv {
            runtime,
            async_connection: mut connection,
            handler: _handler,
            ..
        } = MockEnv::with_client_builder(
            ClusterClient::builder(vec![&*format!("redis://{name}")]).retries(1),
            name,
            move |cmd: &[u8], port| {
                if port == 6380 {
                    respond_startup_two_nodes(name, cmd)?;
                    return Err(parse_redis_value(
                        format!("-MOVED 123 {name}:6379\r\n").as_bytes(),
                    ));
                }

                if contains_slice(cmd, b"PING") {
                    let connect_attempt = ping_attempts_clone.fetch_add(1, Ordering::Relaxed);
                    let past_get_attempts = get_attempts.load(Ordering::Relaxed);
                    // We want connection checks to fail after the first GET attempt, until it retries. Hence, we wait for 5 PINGs -
                    // 1. initial connection,
                    // 2. refresh slots on client creation,
                    // 3. refresh_connections `check_connection` after first GET failed,
                    // 4. refresh_connections `connect_and_check` after first GET failed,
                    // 5. reconnect on 2nd GET attempt.
                    // more than 5 attempts mean that the server reconnects more than once, which is the behavior we're testing against.
                    if past_get_attempts != 1 || connect_attempt > 3 {
                        respond_startup_two_nodes(name, cmd)?;
                    }
                    if connect_attempt > 5 {
                        panic!("Too many pings!");
                    }
                    Err(Err(RedisError::from((
                        ErrorKind::FatalSendError,
                        "mock-io-error",
                    ))))
                } else {
                    respond_startup_two_nodes(name, cmd)?;
                    let past_get_attempts = get_attempts.fetch_add(1, Ordering::Relaxed);
                    // we fail the initial GET request, and after that we'll fail the first reconnect attempt, in the `refresh_connections` attempt.
                    if past_get_attempts == 0 {
                        // Error once with io-error, ensure connection is reestablished w/out calling
                        // other node (i.e., not doing a full slot rebuild)
                        Err(Err(RedisError::from((
                            ErrorKind::FatalSendError,
                            "mock-io-error",
                        ))))
                    } else {
                        Err(Ok(Value::BulkString(b"123".to_vec())))
                    }
                }
            },
        );

        for _ in 0..4 {
            let value = runtime.block_on(
                cmd("GET")
                    .arg("test")
                    .query_async::<_, Option<i32>>(&mut connection),
            );

            assert_eq!(value, Ok(Some(123)));
        }
        // If you need to change the number here due to a change in the cluster, you probably also need to adjust the test.
        // See the PING counts above to explain why 5 is the target number.
        assert_eq!(ping_attempts.load(Ordering::Acquire), 5);
    }

    #[test]
    #[serial_test::serial]
    fn test_async_cluster_periodic_checks_use_management_connection() {
        let cluster = TestClusterContext::new_with_cluster_client_builder(
            3,
            0,
            |builder| {
                builder.periodic_topology_checks(Duration::from_millis(10))
                // Disable the rate limiter to refresh slots immediately on the periodic checks
                .slots_refresh_rate_limit(Duration::from_secs(0), 0)
            },
            false,
        );

        block_on_all(async move {
        let mut connection = cluster.async_connection(None).await;
        let mut client_list = "".to_string();
        let max_requests = 1000;
        let mut i = 0;
        loop {
            if i == max_requests {
                break;
            } else {
                client_list = cmd("CLIENT")
                    .arg("LIST")
                    .query_async::<_, String>(&mut connection)
                    .await
                    .expect("Failed executing CLIENT LIST");
                let mut client_list_parts = client_list.split('\n');
                if client_list_parts
                .any(|line| line.contains(MANAGEMENT_CONN_NAME) && line.contains("cmd=cluster")) 
                && client_list.matches(MANAGEMENT_CONN_NAME).count() == 1 {
                    return Ok::<_, RedisError>(());
                }
            }
            i += 1;
            let _ = sleep(futures_time::time::Duration::from_millis(10)).await;
        }
        panic!("Couldn't find a management connection or the connection wasn't used to execute CLUSTER SLOTS {:?}", client_list);
    })
    .unwrap();
    }

    async fn get_clients_names_to_ids(
        connection: &mut ClusterConnection,
        routing: Option<RoutingInfo>,
    ) -> HashMap<String, String> {
        let mut client_list_cmd = redis::cmd("CLIENT");
        client_list_cmd.arg("LIST");
        let value = match routing {
            Some(routing) => connection.route_command(&client_list_cmd, routing).await,
            None => connection.req_packed_command(&client_list_cmd).await,
        }
        .unwrap();
        let string = String::from_owned_redis_value(value).unwrap();
        string
            .split('\n')
            .filter_map(|line| {
                if line.is_empty() {
                    return None;
                }
                let key_values = line
                    .split(' ')
                    .filter_map(|value| {
                        let mut split = value.split('=');
                        match (split.next(), split.next()) {
                            (Some(key), Some(val)) => Some((key, val)),
                            _ => None,
                        }
                    })
                    .collect::<HashMap<_, _>>();
                match (key_values.get("name"), key_values.get("id")) {
                    (Some(key), Some(val)) if !val.is_empty() => {
                        Some((key.to_string(), val.to_string()))
                    }
                    _ => None,
                }
            })
            .collect()
    }

    async fn kill_connection(killer_connection: &mut ClusterConnection, connection_to_kill: &str) {
        let default_routing = RoutingInfo::SingleNode(SingleNodeRoutingInfo::SpecificNode(
            Route::new(0, SlotAddr::Master),
        ));
        kill_connection_with_routing(killer_connection, connection_to_kill, default_routing).await;
    }

    async fn kill_connection_with_routing(
        killer_connection: &mut ClusterConnection,
        connection_to_kill: &str,
        routing: RoutingInfo,
    ) {
        let mut cmd = redis::cmd("CLIENT");
        cmd.arg("KILL");
        cmd.arg("ID");
        cmd.arg(connection_to_kill);
        // Kill the management connection for the routing node
        assert!(killer_connection.route_command(&cmd, routing).await.is_ok());
    }

    #[test]
    #[serial_test::serial]
    fn test_async_cluster_only_management_connection_is_reconnected_after_connection_failure() {
        // This test will check two aspects:
        // 1. Ensuring that after a disconnection in the management connection, a new management connection is established.
        // 2. Confirming that a failure in the management connection does not impact the user connection, which should remain intact.
        let cluster = TestClusterContext::new_with_cluster_client_builder(
            3,
            0,
            |builder| builder.periodic_topology_checks(Duration::from_millis(10)),
            false,
        );
        block_on_all(async move {
        let mut connection = cluster.async_connection(None).await;
        let _client_list = "".to_string();
        let max_requests = 500;
        let mut i = 0;
        // Set the name of the client connection to 'user-connection', so we'll be able to identify it later on
        assert!(cmd("CLIENT")
            .arg("SETNAME")
            .arg("user-connection")
            .query_async::<_, Value>(&mut connection)
            .await
            .is_ok());
        // Get the client list
        let names_to_ids = get_clients_names_to_ids(&mut connection, Some(RoutingInfo::SingleNode(
            SingleNodeRoutingInfo::SpecificNode(Route::new(0, SlotAddr::Master))))).await;

        // Get the connection ID of 'user-connection'
        let user_conn_id = names_to_ids.get("user-connection").unwrap();
        // Get the connection ID of the management connection
        let management_conn_id = names_to_ids.get(MANAGEMENT_CONN_NAME).unwrap();
        // Get another connection that will be used to kill the management connection
        let mut killer_connection = cluster.async_connection(None).await;
        kill_connection(&mut killer_connection, management_conn_id).await;
        loop {
            // In this loop we'll wait for the new management connection to be established
            if i == max_requests {
                break;
            } else {
                let names_to_ids = get_clients_names_to_ids(&mut connection, Some(RoutingInfo::SingleNode(
                    SingleNodeRoutingInfo::SpecificNode(Route::new(0, SlotAddr::Master))))).await;
                if names_to_ids.contains_key(MANAGEMENT_CONN_NAME) {
                    // A management connection is found
                    let curr_management_conn_id =
                    names_to_ids.get(MANAGEMENT_CONN_NAME).unwrap();
                    let curr_user_conn_id =
                    names_to_ids.get("user-connection").unwrap();
                    // Confirm that the management connection has a new connection ID, and verify that the user connection remains unaffected.
                    if (curr_management_conn_id != management_conn_id)
                        && (curr_user_conn_id == user_conn_id)
                    {
                        return Ok::<_, RedisError>(());
                    }
                } else {
                    i += 1;
                    let _ = sleep(futures_time::time::Duration::from_millis(50)).await;
                    continue;
                }
            }
        }
        panic!(
            "No reconnection of the management connection found, or there was an unwantedly reconnection of the user connections.
            \nprev_management_conn_id={:?},prev_user_conn_id={:?}\nclient list={:?}",
            management_conn_id, user_conn_id, names_to_ids
        );
    })
    .unwrap();
    }

    #[test]
    #[serial_test::serial]
    fn test_async_cluster_dont_route_to_a_random_on_non_key_based_cmd() {
        // This test verifies that non-key-based commands do not get routed to a random node
        // when no connection is found for the given route. Instead, the appropriate error
        // should be raised.
        let name = "test_async_cluster_dont_route_to_a_random_on_non_key_based_cmd";
        let request_counter = Arc::new(AtomicU32::new(0));
        let cloned_req_counter = request_counter.clone();
        let MockEnv {
            runtime,
            async_connection: mut connection,
            handler: _handler,
            ..
        } = MockEnv::with_client_builder(
            ClusterClient::builder(vec![&*format!("redis://{name}")]).retries(1),
            name,
            move |received_cmd: &[u8], _| {
                let slots_config_vec = vec![
                    MockSlotRange {
                        primary_port: 6379,
                        replica_ports: vec![],
                        slot_range: (0_u16..8000_u16),
                    },
                    MockSlotRange {
                        primary_port: 6380,
                        replica_ports: vec![],
                        // Don't cover all slots
                        slot_range: (8001_u16..12000_u16),
                    },
                ];
                respond_startup_with_config(name, received_cmd, Some(slots_config_vec), false)?;
                // If requests are sent to random nodes, they will be caught and counted here.
                request_counter.fetch_add(1, Ordering::Relaxed);
                Err(Ok(Value::Nil))
            },
        );

        runtime
            .block_on(async move {
                let uncovered_slot = 16000;
                let route = redis::cluster_routing::Route::new(
                    uncovered_slot,
                    redis::cluster_routing::SlotAddr::Master,
                );
                let single_node_route =
                    redis::cluster_routing::SingleNodeRoutingInfo::SpecificNode(route);
                let routing = RoutingInfo::SingleNode(single_node_route);
                let res = connection
                    .route_command(&redis::cmd("FLUSHALL"), routing)
                    .await;
                assert!(res.is_err());
                let res_err = res.unwrap_err();
                assert_eq!(
                    res_err.kind(),
                    ErrorKind::ConnectionNotFoundForRoute,
                    "{:?}",
                    res_err
                );
                assert_eq!(cloned_req_counter.load(Ordering::Relaxed), 0);
                Ok::<_, RedisError>(())
            })
            .unwrap();
    }

    #[test]
    #[serial_test::serial]
    fn test_async_cluster_route_to_random_on_key_based_cmd() {
        // This test verifies that key-based commands get routed to a random node
        // when no connection is found for the given route. The command should
        // then be redirected correctly by the server's MOVED error.
        let name = "test_async_cluster_route_to_random_on_key_based_cmd";
        let request_counter = Arc::new(AtomicU32::new(0));
        let cloned_req_counter = request_counter.clone();
        let MockEnv {
            runtime,
            async_connection: mut connection,
            handler: _handler,
            ..
        } = MockEnv::with_client_builder(
            ClusterClient::builder(vec![&*format!("redis://{name}")]),
            name,
            move |received_cmd: &[u8], _| {
                let slots_config_vec = vec![
                    MockSlotRange {
                        primary_port: 6379,
                        replica_ports: vec![],
                        slot_range: (0_u16..8000_u16),
                    },
                    MockSlotRange {
                        primary_port: 6380,
                        replica_ports: vec![],
                        // Don't cover all slots
                        slot_range: (8001_u16..12000_u16),
                    },
                ];
                respond_startup_with_config(name, received_cmd, Some(slots_config_vec), false)?;
                if contains_slice(received_cmd, b"GET") {
                    if request_counter.fetch_add(1, Ordering::Relaxed) == 0 {
                        return Err(parse_redis_value(
                            format!("-MOVED 12182 {name}:6380\r\n").as_bytes(),
                        ));
                    } else {
                        return Err(Ok(Value::SimpleString("bar".into())));
                    }
                }
                panic!("unexpected command {:?}", received_cmd);
            },
        );

        runtime
            .block_on(async move {
                // The keyslot of "foo" is 12182 and it isn't covered by any node, so we expect the
                // request to be routed to a random node and then to be redirected to the MOVED node (2 requests in total)
                let res: String = connection.get("foo").await.unwrap();
                assert_eq!(res, "bar".to_string());
                assert_eq!(cloned_req_counter.load(Ordering::Relaxed), 2);
                Ok::<_, RedisError>(())
            })
            .unwrap();
    }

    #[test]
    #[serial_test::serial]
    fn test_async_cluster_do_not_retry_when_receiver_was_dropped() {
        let name = "test_async_cluster_do_not_retry_when_receiver_was_dropped";
        let cmd = cmd("FAKE_COMMAND");
        let packed_cmd = cmd.get_packed_command();
        let request_counter = Arc::new(AtomicU32::new(0));
        let cloned_req_counter = request_counter.clone();
        let MockEnv {
            runtime,
            async_connection: mut connection,
            ..
        } = MockEnv::with_client_builder(
            ClusterClient::builder(vec![&*format!("redis://{name}")])
                .retries(5)
                .max_retry_wait(2)
                .min_retry_wait(2),
            name,
            move |received_cmd: &[u8], _| {
                respond_startup(name, received_cmd)?;

                if received_cmd == packed_cmd {
                    cloned_req_counter.fetch_add(1, Ordering::Relaxed);
                    return Err(Err((ErrorKind::TryAgain, "seriously, try again").into()));
                }

                Err(Ok(Value::Okay))
            },
        );

        runtime.block_on(async move {
            let err = cmd
                .query_async::<_, Value>(&mut connection)
                .timeout(futures_time::time::Duration::from_millis(1))
                .await
                .unwrap_err();
            assert_eq!(err.kind(), std::io::ErrorKind::TimedOut);

            // we sleep here, to allow the cluster connection time to retry. We expect it won't, but without this
            // sleep the test will complete before the the runtime gave the connection time to retry, which would've made the
            // test pass regardless of whether the connection tries retrying or not.
            sleep(Duration::from_millis(10).into()).await;
        });

        assert_eq!(request_counter.load(Ordering::Relaxed), 1);
    }

    #[cfg(feature = "tls-rustls")]
    mod mtls_test {
        use crate::support::mtls_test::create_cluster_client_from_cluster;
        use redis::ConnectionInfo;

        use super::*;

        #[test]
        #[serial_test::serial]
        fn test_async_cluster_basic_cmd_with_mtls() {
            let cluster = TestClusterContext::new_with_mtls(3, 0);
            block_on_all(async move {
                let client = create_cluster_client_from_cluster(&cluster, true).unwrap();
                let mut connection = client.get_async_connection(None).await.unwrap();
                cmd("SET")
                    .arg("test")
                    .arg("test_data")
                    .query_async(&mut connection)
                    .await?;
                let res: String = cmd("GET")
                    .arg("test")
                    .clone()
                    .query_async(&mut connection)
                    .await?;
                assert_eq!(res, "test_data");
                Ok::<_, RedisError>(())
            })
            .unwrap();
        }

        #[test]
        #[serial_test::serial]
        fn test_async_cluster_should_not_connect_without_mtls_enabled() {
            let cluster = TestClusterContext::new_with_mtls(3, 0);
            block_on_all(async move {
            let client = create_cluster_client_from_cluster(&cluster, false).unwrap();
            let connection = client.get_async_connection(None).await;
            match cluster.cluster.servers.first().unwrap().connection_info() {
                ConnectionInfo {
                    addr: redis::ConnectionAddr::TcpTls { .. },
                    ..
            } => {
                if connection.is_ok() {
                    panic!("Must NOT be able to connect without client credentials if server accepts TLS");
                }
            }
            _ => {
                if let Err(e) = connection {
                    panic!("Must be able to connect without client credentials if server does NOT accept TLS: {e:?}");
                }
            }
            }
            Ok::<_, RedisError>(())
        }).unwrap();
        }
    }
}