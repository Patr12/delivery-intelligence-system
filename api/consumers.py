# consumers.py
from channels.generic.websocket import AsyncJsonWebsocketConsumer
from datetime import datetime

class OrderTrackerConsumer(AsyncJsonWebsocketConsumer):
    async def connect(self):
        await self.accept()

    async def receive_json(self, content):
        order_ids = content.get("order_ids", [])
        # Simulate location
        locations = []
        for idx, oid in enumerate(order_ids):
            locations.append({
                "order_id": oid,
                "lat": -3.3869 + 0.001*idx,
                "lng": 36.6830 + 0.001*idx,
                "last_updated": datetime.utcnow().isoformat()
            })
        await self.send_json({"success": True, "locations": locations})
