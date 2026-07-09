# Human Route Lesson

Route session: `vision_dataset/raw_human_play/session_z_20260619_161751`

## Summary

- Frames/actions recorded: **699**
- Important events detected: **32**
- Use rows: **25**
- Shoot rows: **119**
- Kills gained: **5**
- Health lost: **6.0**
- Start position: `(-416.0, 256.0, 0.0)`
- Final position: `(-368.0, 1308.5, -128.0)`

## What the AI should notice

- Movement is not random: Steven follows a path toward doors, rooms, and the final button.
- Shooting is tied to route safety: when an enemy blocks or threatens the route, shooting helps progress.
- Use is special: it usually means a door, switch, or final button, not a movement action to spam forever.
- Turns matter: when the path bends, the agent should face the next route direction before moving forward.
- Z changes matter: floor height transitions indicate stairs, lifts, drops, or different reachable areas.
- The final use event matters most: successful completion should be saved as high-value training data.

## Timeline of important events

- **spawn** at idx `0` frame `0` pos=(-416.0, 256.0, 0.0) angle=0.0: Start of the route. Establish the first forward direction from spawn.
- **shoot** at idx `105` frame `105` pos=(237.1, 273.8, -65.0) angle=0.0: Shoot because an enemy or threat is on/near the route. This protects progress and prevents dying. `ammo 50->50`
- **shoot** at idx `113` frame `113` pos=(303.9, 274.9, -128.0) angle=0.0: Shoot because an enemy or threat is on/near the route. This protects progress and prevents dying. `ammo 49->49`
- **shoot** at idx `121` frame `121` pos=(375.3, 275.7, -128.0) angle=0.0: Shoot because an enemy or threat is on/near the route. This protects progress and prevents dying. `ammo 49->49`
- **shoot** at idx `129` frame `129` pos=(444.1, 276.1, -128.0) angle=0.0: Shoot because an enemy or threat is on/near the route. This protects progress and prevents dying. `ammo 48->48`
- **shoot** at idx `137` frame `137` pos=(511.8, 276.3, -128.0) angle=0.0: Shoot because an enemy or threat is on/near the route. This protects progress and prevents dying. `ammo 48->48`
- **kill** at idx `138` frame `138` pos=(520.2, 276.3, -128.0) angle=0.0: Enemy was killed here. This confirms shooting was useful for clearing the route. `kills 0->1`
- **shoot** at idx `145` frame `145` pos=(578.9, 276.4, -128.0) angle=3.5: Shoot because an enemy or threat is on/near the route. This protects progress and prevents dying. `ammo 47->47`
- **shoot** at idx `153` frame `153` pos=(645.3, 281.0, -128.0) angle=15.8: Shoot because an enemy or threat is on/near the route. This protects progress and prevents dying. `ammo 46->46`
- **use** at idx `194` frame `194` pos=(835.5, 495.8, -128.0) angle=87.9: Press use here because this is likely a door, switch, or route gate. `final_area=0`
- **shoot** at idx `249` frame `249` pos=(838.6, 573.4, -128.0) angle=93.2: Shoot because an enemy or threat is on/near the route. This protects progress and prevents dying. `ammo 56->56`
- **shoot** at idx `257` frame `257` pos=(836.2, 635.5, -128.0) angle=93.2: Shoot because an enemy or threat is on/near the route. This protects progress and prevents dying. `ammo 55->55`
- **shoot** at idx `265` frame `265` pos=(833.6, 689.6, -128.0) angle=93.2: Shoot because an enemy or threat is on/near the route. This protects progress and prevents dying. `ammo 55->55`
- **shoot** at idx `273` frame `273` pos=(843.5, 694.1, -128.0) angle=93.2: Shoot because an enemy or threat is on/near the route. This protects progress and prevents dying. `ammo 55->55`
- **shoot** at idx `285` frame `285` pos=(885.5, 686.2, -128.0) angle=93.2: Shoot because an enemy or threat is on/near the route. This protects progress and prevents dying. `ammo 55->55`
- **shoot** at idx `293` frame `293` pos=(896.9, 684.4, -128.0) angle=93.2: Shoot because an enemy or threat is on/near the route. This protects progress and prevents dying. `ammo 55->55`
- **shoot** at idx `301` frame `301` pos=(901.5, 693.9, -128.0) angle=93.2: Shoot because an enemy or threat is on/near the route. This protects progress and prevents dying. `ammo 55->55`
- **kill** at idx `302` frame `302` pos=(901.7, 697.5, -128.0) angle=93.2: Enemy was killed here. This confirms shooting was useful for clearing the route. `kills 1->2`
- **shoot** at idx `309` frame `309` pos=(901.7, 733.2, -128.0) angle=94.9: Shoot because an enemy or threat is on/near the route. This protects progress and prevents dying. `ammo 55->55`
- **use** at idx `368` frame `368` pos=(590.1, 1012.1, -128.0) angle=167.0: Press use here because this is likely a door, switch, or route gate. `final_area=0`
- **shoot** at idx `443` frame `443` pos=(291.6, 1171.6, -128.0) angle=72.1: Shoot because an enemy or threat is on/near the route. This protects progress and prevents dying. `ammo 95->95`
- **kill** at idx `447` frame `447` pos=(283.8, 1201.6, -128.0) angle=61.5: Enemy was killed here. This confirms shooting was useful for clearing the route. `kills 2->3`
- **shoot** at idx `502` frame `502` pos=(261.6, 1550.9, -128.0) angle=168.8: Shoot because an enemy or threat is on/near the route. This protects progress and prevents dying. `ammo 105->105`
- **kill** at idx `506` frame `506` pos=(232.1, 1561.6, -128.0) angle=168.8: Enemy was killed here. This confirms shooting was useful for clearing the route. `kills 3->4`
- **shoot** at idx `542` frame `542` pos=(6.1, 1603.9, -128.0) angle=198.6: Shoot because an enemy or threat is on/near the route. This protects progress and prevents dying. `ammo 105->105`
- **hurt** at idx `545` frame `545` pos=(-16.5, 1597.7, -128.0) angle=198.6: Health dropped here. Treat this as danger; shooting or faster movement may be needed. `health 104->98`
- **shoot** at idx `550` frame `550` pos=(-51.8, 1587.5, -128.0) angle=202.1: Shoot because an enemy or threat is on/near the route. This protects progress and prevents dying. `ammo 105->105`
- **kill** at idx `554` frame `554` pos=(-80.8, 1577.4, -128.0) angle=210.9: Enemy was killed here. This confirms shooting was useful for clearing the route. `kills 4->5`
- **use** at idx `587` frame `587` pos=(-229.1, 1416.0, -128.0) angle=253.1: Press use here because this appears to be the final door/button needed to complete the level. `final_area=1`
- **use** at idx `669` frame `669` pos=(-367.9, 1310.1, -128.0) angle=167.0: Press use here because this appears to be the final door/button needed to complete the level. `final_area=1`
- **use** at idx `697` frame `697` pos=(-368.0, 1309.5, -128.0) angle=203.9: Press use here because this appears to be the final door/button needed to complete the level. `final_area=1`
- **finish** at idx `698` frame `698` pos=(-368.0, 1308.5, -128.0) angle=203.9: End of the demonstration. Treat the final position/actions as the completion target.

## Training lesson

This route should be used as a successful demonstration only if the run completes the level. The agent should learn the sequence: follow route, clear route enemies, use doors/switches briefly, cross after use, then face and press the final button.
