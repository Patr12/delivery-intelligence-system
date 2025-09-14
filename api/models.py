from django.db import models

# Create your models here.
# models.py
from django.db import models

class Courier(models.Model):
    name = models.CharField(max_length=100)
    vehicle_type = models.CharField(max_length=50, choices=[
        ('motorcycle', 'Motorcycle'),
        ('car', 'Car'),
        ('truck', 'Truck')
    ])
    efficiency = models.FloatField(default=1.0)  # 1.0 = normal, higher = better
    current_lat = models.FloatField(null=True, blank=True)
    current_lng = models.FloatField(null=True, blank=True)
    is_available = models.BooleanField(default=True)
    
    def __str__(self):
        return f"{self.name} ({self.vehicle_type})"


class Order(models.Model):
    order_id = models.CharField(max_length=50, unique=True)
    pickup_lat = models.FloatField()
    pickup_lng = models.FloatField()
    drop_lat = models.FloatField()
    drop_lng = models.FloatField()
    status = models.CharField(max_length=20, choices=[
        ("pending", "Pending"), ("in_transit", "In Transit"), ("delivered", "Delivered")
    ], default="pending")
    courier = models.ForeignKey(Courier, null=True, blank=True, on_delete=models.SET_NULL)
    created_at = models.DateTimeField(auto_now_add=True)
    delivered_at = models.DateTimeField(null=True, blank=True)
    co2_emission_kg = models.FloatField(null=True, blank=True)

    def trip_km(self):
        # haversine formula or stored field
        return 5.0  # placeholder
