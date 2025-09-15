from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
import pandas as pd
import requests
from django.views.decorators.csrf import csrf_exempt

import numpy as np
from django.conf import settings
from django.db.models import Count, Avg, F, ExpressionWrapper, DurationField
from django.utils.timezone import now
from geopy.distance import geodesic
from .models import Courier, Order
from .ml import model, encoders
from .utils import hybrid_anomaly_detection, normalize_columns, coerce_times, add_features, featurize_for_model, haversine_km, greedy_assign, clean_data



# --- Dashboard ---
def dashboard(request):
    return render(request, "api/dashboard.html")
def route_search_page(request):
    return render(request, "api/route_co2.html")
def detect_anomalies_page(request):
    return render(request, "api/detect_anomalies_page.html")

# --- Utility Functions ---
def parse_datetime_safe(dt_str):
    """Safely parse datetime string to datetime object"""
    from datetime import datetime
    if not dt_str:
        return None
    try:
        if "T" in dt_str:  # ISO format from HTML input
            return datetime.fromisoformat(dt_str)
        else:  # fallback MM-DD HH:MM:SS (add year)
            return datetime.strptime(f"{now().year}-{dt_str}", "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None

def coerce_times(df):
    """Convert datetime columns with proper handling for both ISO and MM-DD formats"""
    datetime_columns = ['receipt_time', 'sign_time']
    
    for c in datetime_columns:
        if c in df.columns:
            # Try ISO format first (from HTML input), then fallback to MM-DD format
            df[c] = pd.to_datetime(df[c], errors='coerce')
            
            # For any remaining NaT values, try the MM-DD format
            if df[c].isna().any():
                # Try adding current year for MM-DD format
                current_year = pd.Timestamp.now().year
                df[c] = df[c].fillna(pd.to_datetime(
                    f"{current_year}-" + df[c].astype(str), 
                    format="%Y-%m-%d %H:%M:%S", 
                    errors="coerce"
                ))
    
    return df

def prepare_orders(orders):
    prepared = []
    for o in orders:
        try:
            # Handle datetime inputs from HTML form
            receipt_time = o.get("receipt_time")
            sign_time = o.get("sign_time")
            
            # Convert ISO format (from HTML) to consistent format
            if receipt_time and "T" in receipt_time:
                receipt_time = receipt_time.replace("T", " ")
            
            if sign_time and "T" in sign_time:
                sign_time = sign_time.replace("T", " ")
            
            # Create order dictionary
            order_dict = {
                "order_id": o.get("order_id", ""),
                "from_city_name": o.get("from_city_name", ""),
                "poi_lat": float(o.get("poi_lat", 0)),
                "poi_lng": float(o.get("poi_lng", 0)),
                "receipt_time": receipt_time,
                "receipt_lat": float(o.get("receipt_lat", 0)),
                "receipt_lng": float(o.get("receipt_lng", 0)),
                "sign_time": sign_time,
                "sign_lat": float(o.get("sign_lat", 0)),
                "sign_lng": float(o.get("sign_lng", 0)),
                "typecode": str(o.get("typecode", "")),
                "aoi_id": str(o.get("aoi_id", ""))
            }
            
            # Add trip_km if provided from frontend
            if 'trip_km' in o:
                order_dict['trip_km'] = float(o['trip_km'])
                
            prepared.append(order_dict)
        except Exception as e:
            print(f"Error preparing order: {e}")
            continue
    
    df = pd.DataFrame(prepared)
    return df

@csrf_exempt
@api_view(["POST"])
def predict_eta(request):
    orders = request.data.get("orders", [])
    if not orders:
        return Response({"success": False, "error": "No orders provided"}, status=status.HTTP_400_BAD_REQUEST)

    try:
        print("📦 Received orders:", orders)
        
        df = prepare_orders(orders)
        print("📊 Prepared DataFrame shape:", df.shape)
        print("📊 DataFrame columns:", df.columns.tolist())
        
        if df.empty:
            return Response({"success": False, "error": "Invalid order data"}, status=status.HTTP_400_BAD_REQUEST)

        df = normalize_columns(df)
        df = coerce_times(df)
        
        # Calculate trip_km if not present
        if 'trip_km' not in df.columns:
            from .utils import haversine_km
            df["trip_km"] = haversine_km(
                df["poi_lat"].values, 
                df["poi_lng"].values,
                df["receipt_lat"].values, 
                df["receipt_lng"].values
            )
        
        # ADD THIS: Add speed_km_min with default value for prediction
        if 'speed_km_min' not in df.columns:
            df['speed_km_min'] = 0.3  # default average speed (km/min) = 18 km/h
        
        df = add_features(df)
        X, _, _ = featurize_for_model(df, fit_encoders=False, encoders=encoders)

        preds = model.predict(X)
        predictions = [{"order_id": str(r["order_id"]), "eta_minutes": float(p)} for r, p in zip(df.to_dict(orient="records"), preds)]
        return Response({"success": True, "predictions": predictions})
    except Exception as e:
        print("❌ Error in predict_eta:", str(e))
        return Response({"success": False, "error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)



# --- Predict ETA Smart ---
@api_view(["POST"])
def predict_eta_smart(request):
    orders = request.data.get("orders", [])
    if not orders:
        return Response({"success": False, "error": "No orders provided"}, status=status.HTTP_400_BAD_REQUEST)

    try:
        df = prepare_orders(orders)
        if df.empty:
            return Response({"success": False, "error": "Invalid order data"}, status=status.HTTP_400_BAD_REQUEST)

        df = normalize_columns(df)
        df = coerce_times(df)
        df = add_features(df)

        # Weather factor
        try:
            lat, lng = df["poi_lat"].iloc[0], df["poi_lng"].iloc[0]
            weather = requests.get(
                f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lng}&appid={settings.OWM_KEY}"
            ).json()
            df["rain_intensity"] = weather.get("rain", {}).get("1h", 0)
        except:
            df["rain_intensity"] = 0

        # Traffic factor (placeholder)
        df["traffic_index"] = 1.2

        X, _, _ = featurize_for_model(df, fit_encoders=False, encoders=encoders)
        preds = model.predict(X)
        predictions = [{"order_id": str(r["order_id"]), "eta_minutes": float(p)} for r, p in zip(df.to_dict(orient="records"), preds)]
        return Response({"success": True, "predictions": predictions})
    except Exception as e:
        return Response({"success": False, "error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# --- Track Orders ---
@api_view(["GET"])
def track_orders(request):
    order_ids = request.GET.getlist("order_ids")
    if not order_ids:
        return Response({"success": False, "error": "No order_ids provided"}, status=status.HTTP_400_BAD_REQUEST)
    
    locations = []
    for idx, oid in enumerate(order_ids):
        # Simulate movement based on index instead of trying to convert string to int
        locations.append({
            "order_id": oid,
            "lat": -3.3869 + 0.001*idx,  
            "lng": 36.6830 + 0.001*idx,
            "last_updated": now().isoformat()
        })
    return Response({"success": True, "locations": locations})


# --- Optimize Route ---
@api_view(["POST"])
def optimize_route(request):
    orders = request.data.get("orders", [])
    couriers = request.data.get("couriers", [])
    if not orders or not couriers:
        return Response({"success": False, "error": "Orders or Couriers missing"}, status=status.HTTP_400_BAD_REQUEST)

    orders_df = pd.DataFrame(orders)
    couriers_df = pd.DataFrame(couriers)

    optimized = greedy_assign(orders_df, couriers_df)
    return Response({"success": True, "assignments": optimized.to_dict(orient="records")})



@api_view(["POST"])
def carbon_footprint(request):
    orders = request.data.get("orders", [])
    if not orders:
        return Response({"success": False, "error": "No orders provided"}, status=status.HTTP_400_BAD_REQUEST)

    df = pd.DataFrame(orders)
    df = normalize_columns(df)
    df = add_features(df)

    # assume courier type = motorbike
    factor = 0.12  # kg CO2 per km
    df["co2_emission_kg"] = df["trip_km"] * factor

    emissions = [{"order_id": str(row["order_id"]), "co2_emission_kg": round(em, 2)} 
                 for row, em in zip(df.to_dict(orient="records"), df["co2_emission_kg"])]

    return Response({"success": True, "emissions": emissions})

@api_view(["GET"])
def courier_performance(request):
    couriers = Courier.objects.annotate(
        total_orders=Count("order"),
        avg_delay=Avg(ExpressionWrapper(F("order__delivered_at") - F("order__created_at"), output_field=DurationField()))
    ).values("id", "name", "vehicle_type", "efficiency", "total_orders", "avg_delay")

    leaderboard = []
    for c in couriers:
        leaderboard.append({
            "id": c["id"],
            "name": c["name"],
            "vehicle_type": c["vehicle_type"],
            "total_orders": c["total_orders"],
            "avg_delay_min": (c["avg_delay"].total_seconds() / 60) if c["avg_delay"] else None,
            "efficiency": c["efficiency"]
        })

    leaderboard = sorted(leaderboard, key=lambda x: (-x["total_orders"], x["avg_delay_min"] or 0))
    return Response({"success": True, "leaderboard": leaderboard})

@api_view(["GET"])
def system_insights(request):
    today = now().date()
    orders_today = Order.objects.filter(created_at__date=today)

    total_orders_today = orders_today.count()
    avg_eta = orders_today.exclude(delivered_at=None).annotate(
        eta=ExpressionWrapper(F("delivered_at") - F("created_at"), output_field=DurationField())
    ).aggregate(avg_eta=Avg("eta"))["avg_eta"]

    total_emissions = orders_today.aggregate(total=Avg("co2_emission_kg"))["total"]

    best_courier = Courier.objects.annotate(total_orders=Count("order")).order_by("-total_orders").first()

    insights = {
        "total_orders_today": total_orders_today,
        "avg_eta_minutes": (avg_eta.total_seconds() / 60) if avg_eta else None,
        "total_co2_emissions_kg": total_emissions,
        "best_courier": {
            "id": best_courier.id,
            "name": best_courier.name,
        } if best_courier else None
    }
    return Response({"success": True, "insights": insights})
# --- HTML Page: Predict ETA + Track Orders ---


def predict_eta_page(request):
    orders = []
    predictions = []
    error = None

    def safe_float(val, default=None):
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    if request.method == "POST":
        try:
            # Get form inputs
            order_ids = request.POST.getlist("order_id")
            from_cities = request.POST.getlist("from_city_name")
            poi_lats = request.POST.getlist("poi_lat")
            poi_lngs = request.POST.getlist("poi_lng")
            receipt_lats = request.POST.getlist("receipt_lat")
            receipt_lngs = request.POST.getlist("receipt_lng")
            typecodes = request.POST.getlist("typecode")

            # Build orders
            for i in range(len(order_ids)):
                poi_lat = safe_float(poi_lats[i])
                poi_lng = safe_float(poi_lngs[i])
                receipt_lat = safe_float(receipt_lats[i])
                receipt_lng = safe_float(receipt_lngs[i])

                if None in [poi_lat, poi_lng, receipt_lat, receipt_lng]:
                    continue

                orders.append({
                    "order_id": order_ids[i],
                    "from_city_name": from_cities[i],
                    "poi_lat": poi_lat,
                    "poi_lng": poi_lng,
                    "receipt_lat": receipt_lat,
                    "receipt_lng": receipt_lng,
                    "typecode": typecodes[i],
                })

            if not orders:
                error = "No valid orders submitted"
            else:
                # --- Pipeline ---
                df = prepare_orders(orders)
                df = normalize_columns(df)
                df = coerce_times(df)
                df = add_features(df)

                # Replace inf and NaN
                df.replace([np.inf, -np.inf], np.nan, inplace=True)

                # Fill numeric columns with safe defaults
                df["trip_km"] = df.get("trip_km", pd.Series([1.0]*len(df))).fillna(1.0)
                df["speed_km_min"] = df.get("speed_km_min", pd.Series([0.3]*len(df))).fillna(0.3)

                # Fill integer columns with 0 and cast safely
                for col in ["hour", "dow", "is_weekend"]:
                    if col not in df.columns:
                        df[col] = 0
                    else:
                        df[col] = df[col].fillna(0)
                    df[col] = df[col].astype(np.int8)

                # Weather factor
                try:
                    lat, lng = df["poi_lat"].iloc[0], df["poi_lng"].iloc[0]
                    weather = requests.get(
                        f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lng}&appid={settings.OWM_KEY}"
                    ).json()
                    df["rain_intensity"] = weather.get("rain", {}).get("1h", 0)
                except Exception:
                    df["rain_intensity"] = 0

                df["traffic_index"] = 1.2

                # Prediction
                X, _, _ = featurize_for_model(df, fit_encoders=False, encoders=encoders)
                preds = model.predict(X)

                predictions = [
                    {"order_id": str(r["order_id"]), "eta_minutes": float(p)}
                    for r, p in zip(df.to_dict(orient="records"), preds)
                ]

        except Exception as e:
            error = str(e)

    context = {
        "orders": orders,
        "predictions": predictions,
        "error": error,
    }
    return render(request, "api/predict_eta.html", context)


def system_insights_page(request):
    today = now().date()
    orders_today = Order.objects.filter(created_at__date=today)

    total_orders_today = orders_today.count()
    avg_eta = orders_today.exclude(delivered_at=None).annotate(
        eta=ExpressionWrapper(F("delivered_at") - F("created_at"), output_field=DurationField())
    ).aggregate(avg_eta=Avg("eta"))["avg_eta"]

    total_emissions = orders_today.aggregate(total=Avg("co2_emission_kg"))["total"]

    best_courier = Courier.objects.annotate(total_orders=Count("order")).order_by("-total_orders").first()

    context = {
        "total_orders_today": total_orders_today,
        "avg_eta_minutes": (avg_eta.total_seconds() / 60) if avg_eta else None,
        "total_co2_emissions_kg": total_emissions,
        "best_courier": best_courier.name if best_courier else None
    }

    return render(request, "api/system_insights.html", context)

# --- Anomaly Detection API ---
@csrf_exempt
@api_view(["POST"])
def detect_anomalies(request):
    try:
        print("DEBUG request.data:", request.data)

        orders = request.data.get("orders", [])
        if not orders:
            return Response(
                {"success": False, "error": "⚠️ Hakuna maagizo yaliyotumwa"},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Prepare orders
        prepared_orders = []
        for order in orders:
            order_data = {
                "order_id": order.get("order_id", ""),
                "from_city_name": order.get("from_city_name", ""),
                "poi_lat": float(order.get("poi_lat", 0)),
                "poi_lng": float(order.get("poi_lng", 0)),
                "receipt_lat": float(order.get("receipt_lat", 0)),
                "receipt_lng": float(order.get("receipt_lng", 0)),
                "typecode": order.get("typecode", ""),
                "num_items": int(order.get("num_items", 1))
            }
            if order.get("receipt_time"):
                order_data["receipt_time"] = order["receipt_time"]
            if order.get("sign_time"):
                order_data["sign_time"] = order["sign_time"]

            prepared_orders.append(order_data)

        df = pd.DataFrame(prepared_orders)
        print("DEBUG raw DataFrame:", df.head())

        if df.empty:
            return Response(
                {"success": False, "error": "❌ Data ya maagizo si sahihi"},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Process
        df = normalize_columns(df)
        df = coerce_times(df)
        df = add_features(df)
        print("DEBUG with features:", df.head())

        feature_cols = [
            c for c in ["trip_km", "delivery_duration_min", "hour", "dow", "is_weekend", "speed_km_min"]
            if c in df.columns
        ]

        if len(feature_cols) < 2:
            return Response(
                {"success": False, "error": "❌ Taarifa haitoshi kuchambua anomalies"},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Run hybrid anomaly detection
        df = hybrid_anomaly_detection(df, feature_cols)

        anomalies = []
        for _, row in df.iterrows():
            anomaly_data = {
                "order_id": str(row.get("order_id", "")),
                "from_city": row.get("from_city_name", ""),
                "trip_km": float(row.get("trip_km", 0)),
                "duration_min": float(row.get("delivery_duration_min", 0)),
                "anomaly": int(row.get("anomaly", 0)),
                "anomaly_score": float(row.get("anomaly_score", 0)),
                "flags": {
                    "duration_flag": int(row.get("flag_duration", 0)),
                    "items_flag": int(row.get("flag_items", 0)),
                    "iforest_flag": int(row.get("flag_iforest", 0))
                }
            }
            anomalies.append(anomaly_data)

        return Response({"success": True, "anomalies": anomalies}, status=status.HTTP_200_OK)

    except Exception as e:
        return Response(
            {"success": False, "error": f"❌ Tatizo: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )