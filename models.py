"""
models.py — Pydantic data models for ShopWave.
"""
from pydantic import BaseModel
from typing import Optional
from datetime import date


class Address(BaseModel):
    street: str
    city: str
    state: str
    zip: str


class Customer(BaseModel):
    customer_id: str
    name: str
    email: str
    phone: str
    tier: str                   # standard | premium | vip
    member_since: date
    total_orders: int
    total_spent: float
    address: Address
    notes: str


class Order(BaseModel):
    order_id: str
    customer_id: str
    product_id: str
    quantity: int
    amount: float
    status: str                 # processing | shipped | delivered | cancelled
    order_date: date
    delivery_date: Optional[date] = None
    return_deadline: Optional[date] = None
    refund_status: Optional[str] = None
    notes: str


class Product(BaseModel):
    product_id: str
    name: str
    category: str
    price: float
    warranty_months: int
    return_window_days: int
    returnable: bool
    notes: str
