# Generated by Django 4.2.16 on 2024-10-23 16:04

import datetime
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('predict', '0017_remove_watchlist_added_at_alter_watchlist_ticker'),
    ]

    operations = [
        migrations.AddField(
            model_name='watchlist',
            name='added_at',
            field=models.DateTimeField(auto_now_add=True, default=datetime.datetime(2024, 10, 23, 16, 4, 35, 288515, tzinfo=datetime.timezone.utc)),
            preserve_default=False,
        ),
        migrations.AlterField(
            model_name='watchlist',
            name='ticker',
            field=models.CharField(max_length=15),
        ),
    ]