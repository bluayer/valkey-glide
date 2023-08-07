import * as net from "net";
import { createCustomCommand } from "./Commands";
import { connection_request, redis_request } from "./ProtobufMessage";
import { ConnectionOptions, RedisClient, ReturnType } from "./RedisClient";

export type SlotIdTypes = {
    type: "primarySlotId" | "replicaSlotId";
    id: number;
};

export type SlotKeyTypes = {
    type: "primarySlotKey" | "replicaSlotKey";
    key: string;
};

export type Routes =
    /// Route request to all primary nodes.
    | "allPrimaries"
    /// Route request to all nodes.
    | "allNodes"
    /// Route request to a random node.
    | "randomNode"
    /// Route request to the node that contains the slot with the given id.
    | SlotIdTypes
    /// Route request to the node that contains the slot that the given key matches.
    | SlotKeyTypes;

function toProtobufRoute(
    route: Routes | undefined
): redis_request.Routes | undefined {
    if (route === undefined) {
        return undefined;
    }
    if (route === "allPrimaries") {
        return redis_request.Routes.create({
            simpleRoutes: redis_request.SimpleRoutes.AllPrimaries,
        });
    } else if (route === "allNodes") {
        return redis_request.Routes.create({
            simpleRoutes: redis_request.SimpleRoutes.AllNodes,
        });
    } else if (route === "randomNode") {
        return redis_request.Routes.create({
            simpleRoutes: redis_request.SimpleRoutes.Random,
        });
    } else if (route.type === "primarySlotKey") {
        return redis_request.Routes.create({
            slotKeyRoute: redis_request.SlotKeyRoute.create({
                slotType: redis_request.SlotTypes.Primary,
                slotKey: route.key,
            }),
        });
    } else if (route.type === "replicaSlotKey") {
        return redis_request.Routes.create({
            slotKeyRoute: redis_request.SlotKeyRoute.create({
                slotType: redis_request.SlotTypes.Replica,
                slotKey: route.key,
            }),
        });
    } else if (route.type === "primarySlotId") {
        return redis_request.Routes.create({
            slotKeyRoute: redis_request.SlotIdRoute.create({
                slotType: redis_request.SlotTypes.Primary,
                slotId: route.id,
            }),
        });
    } else if (route.type === "replicaSlotId") {
        return redis_request.Routes.create({
            slotKeyRoute: redis_request.SlotIdRoute.create({
                slotType: redis_request.SlotTypes.Replica,
                slotId: route.id,
            }),
        });
    }
}

export class RedisClusterClient extends RedisClient {
    protected createClientRequest(
        options: ConnectionOptions
    ): connection_request.IConnectionRequest {
        const configuration = super.createClientRequest(options);
        configuration.clusterModeEnabled = true;
        return configuration;
    }

    public static async createClient(
        options: ConnectionOptions
    ): Promise<RedisClusterClient> {
        return await super.createClientInternal(
            options,
            (socket: net.Socket, options?: ConnectionOptions) =>
                new RedisClusterClient(socket, options)
        );
    }

    static async __createClient(
        options: ConnectionOptions,
        connectedSocket: net.Socket
    ): Promise<RedisClusterClient> {
        return super.__createClientInternal(
            options,
            connectedSocket,
            (socket, options) => new RedisClusterClient(socket, options)
        );
    }

    /** Executes a single command, without checking inputs. Every part of the command, including subcommands,
     *  should be added as a separate value in args.
     *  The command will be routed automatically, unless `route` was provided, in which case the client will
     *  initially try to route the command to the nodes defined by `route`.
     *
     * @example
     * Returns a list of all pub/sub clients on all primary nodes
     * ```ts
     * connection.customCommand("CLIENT", ["LIST","TYPE", "PUBSUB"], "allPrimaries")
     * ```
     */
    public customCommand(
        commandName: string,
        args: string[],
        route?: Routes
    ): Promise<ReturnType> {
        const command = createCustomCommand(commandName, args);
        return super.createWritePromise(command, toProtobufRoute(route));
    }
}
