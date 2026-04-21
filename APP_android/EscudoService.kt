package com.example.detector_spam

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.Build
import android.os.IBinder
import android.provider.Telephony
import androidx.core.app.NotificationCompat

class EscudoService : Service() {

    private lateinit var smsReceiver: BroadcastReceiver

    override fun onCreate() {
        super.onCreate()

        // 1. Creamos la notificación que mantiene viva la app
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel("ESCUDO_ACTIVO", "Protección Activa", NotificationManager.IMPORTANCE_LOW)
            val manager = getSystemService(NotificationManager::class.java)
            manager.createNotificationChannel(channel)
        }

        val notification = NotificationCompat.Builder(this, "ESCUDO_ACTIVO")
            .setContentTitle("Detector Spam Activo 🛡️")
            .setContentText("Protegiendo tu bandeja de entrada en tiempo real...")
            .setSmallIcon(android.R.drawable.stat_notify_chat)
            .setOngoing(true)
            .build()

        // Iniciamos el servicio en primer plano
        startForeground(1001, notification)

        // 2. REGISTRAMOS EL RECEPTOR DINÁMICAMENTE (Esto es lo que burla a Android)
        smsReceiver = Smsrecibidor()
        val filter = IntentFilter(Telephony.Sms.Intents.SMS_RECEIVED_ACTION)
        // Le damos máxima prioridad
        filter.priority = 999

        // Registramos el receptor ligado a este servicio inmortal
        registerReceiver(smsReceiver, filter)
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        // START_STICKY le dice a Android: "Si me matas por falta de RAM, revíveme en cuanto puedas"
        return START_STICKY
    }

    override fun onDestroy() {
        super.onDestroy()
        // Cuando apagamos el escudo, dejamos de escuchar
        try {
            unregisterReceiver(smsReceiver)
        } catch (e: Exception) {
            // Ignorar si ya estaba desregistrado
        }
    }

    override fun onBind(intent: Intent?): IBinder? {
        return null
    }
}