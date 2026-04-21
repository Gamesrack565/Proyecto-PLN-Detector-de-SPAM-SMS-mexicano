package com.example.detector_spam

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.os.Build
import android.provider.Telephony
import android.util.Log
import androidx.core.app.NotificationCompat

class Smsrecibidor : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action == Telephony.Sms.Intents.SMS_RECEIVED_ACTION) {

            val messages = Telephony.Sms.Intents.getMessagesFromIntent(intent)

            // Tomamos el primer mensaje de la lista (por si llegan varios de golpe)
            val sms = messages.firstOrNull() ?: return

            val remitente = sms.displayOriginatingAddress ?: "Desconocido"
            val cuerpo = sms.displayMessageBody ?: ""

            Log.d("SpamDetector", "SMS Interceptado: $remitente. Preparando notificación...")

            mostrarNotificacionParaAnalizar(context, remitente, cuerpo)
        }
    }

    private fun mostrarNotificacionParaAnalizar(context: Context, remitente: String, cuerpo: String) {
        val notificationManager = context.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel("CANAL_SPAM", "Alertas de Spam", NotificationManager.IMPORTANCE_HIGH)
            notificationManager.createNotificationChannel(channel)
        }

        // 1. Creamos un Intent que abrirá tu MainActivity
        val intentApp = Intent(context, MainActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK
            // Le pasamos los datos del SMS y una bandera para saber que venimos de una notificación
            putExtra("nuevo_sms_remitente", remitente)
            putExtra("nuevo_sms_cuerpo", cuerpo)
            putExtra("preguntar_analisis", true)
        }

        // 2. Envolvemos el Intent en un PendingIntent para la notificación
        val pendingIntent = PendingIntent.getActivity(
            context,
            System.currentTimeMillis().toInt(), // ID único
            intentApp,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        // 3. Construimos la notificación
        val builder = NotificationCompat.Builder(context, "CANAL_SPAM")
            .setSmallIcon(android.R.drawable.stat_notify_chat)
            .setContentTitle("📩 Alerta de SMS sospechoso")
            .setContentText("Toca aquí para analizar el mensaje de $remitente")
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            .setAutoCancel(true) // Se borra al tocarla
            .setContentIntent(pendingIntent) // Al tocarla, ejecuta el PendingIntent

        notificationManager.notify(System.currentTimeMillis().toInt(), builder.build())
    }
}