from django.contrib import admin

# Register your models here.
# admin.py
from django.contrib import admin
from .models import Courier, Order

@admin.register(Courier)
class CourierAdmin(admin.ModelAdmin):
    list_display = ("name", "vehicle_type", "efficiency", "is_available", "current_lat", "current_lng")
    list_filter = ("vehicle_type", "is_available")
    search_fields = ("name",)
    ordering = ("-efficiency", "name")


@admin.register(Order)
class OrderAdmin(admin.ModelAdmin):
    list_display = ("order_id", "status", "courier", "created_at", "delivered_at", "co2_emission_kg", "trip_km")
    list_filter = ("status", "courier")
    search_fields = ("order_id",)
    ordering = ("-created_at",)

    readonly_fields = ("trip_km",)
    
    # Optional: customize trip_km display if using placeholder method
    def trip_km(self, obj):
        return obj.trip_km()
    trip_km.short_description = "Trip (km)"
