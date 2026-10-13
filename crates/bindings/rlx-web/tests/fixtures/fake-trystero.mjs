// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Trystero 0.26's shape, as documented in @trystero-p2p/core's types:
//   joinRoom(config, roomId, callbacks?) -> Room
//   Room.makeAction(ns) -> { send, onMessage, … }   // object, not a tuple
//   Room.onPeerJoin / onPeerLeave                    // properties to assign
//
// The real module cannot run outside a browser (no RTCPeerConnection), so this
// stands in for it to pin the contract `cluster-p2p.js` is written against.

export const selfId = "fake-self";

/// Every room created, for assertions.
export const rooms = [];

export function joinRoom(config, roomId, callbacks) {
  const peers = {};
  const actions = new Map();
  const room = {
    config,
    roomId,
    callbacks,
    sent: [],
    onPeerJoin: null,
    onPeerLeave: null,
    makeAction(namespace) {
      const action = {
        namespace,
        onMessage: null,
        onReceiveProgress: null,
        onReceive: null,
        send: async (data) => {
          room.sent.push(data);
        },
      };
      actions.set(namespace, action);
      return action;
    },
    getPeers: () => peers,
    leave: async () => {
      room.left = true;
    },
    // Test handles for driving the fake.
    _deliver(namespace, data) {
      actions.get(namespace)?.onMessage?.(data, { peerId: "peer-1" });
    },
    _addPeer(id) {
      peers[id] = {};
      room.onPeerJoin?.(id);
    },
    _dropPeer(id) {
      delete peers[id];
      room.onPeerLeave?.(id);
    },
    _failJoin(error, peerId = "XmvyuAs4iB96l2rzxekh") {
      callbacks?.onJoinError?.({ error, peerId, appId: config.appId, roomId });
    },
  };
  rooms.push(room);
  return room;
}
