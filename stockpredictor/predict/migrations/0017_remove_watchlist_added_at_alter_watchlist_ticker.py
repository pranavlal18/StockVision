# Generated by Django 4.2.16 on 2024-10-23 15:34

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('predict', '0016_watchlist'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='watchlist',
            name='added_at',
        ),
        migrations.AlterField(
            model_name='watchlist',
            name='ticker',
            field=models.CharField(max_length=10),
        ),
    ]
