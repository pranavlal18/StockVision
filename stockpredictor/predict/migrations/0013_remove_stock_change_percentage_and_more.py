# Generated by Django 4.2.16 on 2024-10-23 12:20

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('predict', '0012_stock_analystrating'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='stock',
            name='change_percentage',
        ),
        migrations.RemoveField(
            model_name='stock',
            name='current_price',
        ),
        migrations.AlterField(
            model_name='stock',
            name='ticker',
            field=models.CharField(max_length=15),
        ),
    ]
