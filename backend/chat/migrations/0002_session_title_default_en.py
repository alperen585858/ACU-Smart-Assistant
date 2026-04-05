from django.db import migrations, models


def forwards(apps, schema_editor):
    ChatSession = apps.get_model("chat", "ChatSession")
    ChatSession.objects.filter(title="Yeni sohbet").update(title="New chat")


class Migration(migrations.Migration):
    dependencies = [
        ("chat", "0001_initial"),
    ]

    operations = [
        migrations.RunPython(forwards, migrations.RunPython.noop),
        migrations.AlterField(
            model_name="chatsession",
            name="title",
            field=models.CharField(default="New chat", max_length=500),
        ),
    ]
