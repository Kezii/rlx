// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// The pre-0.26 shape: `makeAction` returns `[send, receive]` and `onPeerJoin`
// is a registration function. `?p2p=` can point at such a version, so the
// adapter accepts it.

export const selfId = "legacy-self";
export const rooms = [];

export function joinRoom(config, roomId) {
  const peers = {};
  let receive = null;
  const room = {
    config,
    roomId,
    sent: [],
    makeAction: () => [
      (data) => {
        room.sent.push(data);
      },
      (cb) => {
        receive = cb;
      },
    ],
    onPeerJoin: (cb) => {
      room._join = cb;
    },
    onPeerLeave: (cb) => {
      room._leave = cb;
    },
    getPeers: () => peers,
    leave: () => {
      room.left = true;
    },
    _deliver: (data) => receive?.(data, "peer-1"),
    _addPeer: (id) => {
      peers[id] = {};
      room._join?.(id);
    },
  };
  rooms.push(room);
  return room;
}
