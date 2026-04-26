package com.example.detector_spam

import android.Manifest
import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.content.SharedPreferences
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.SystemBarStyle
import androidx.compose.animation.Crossfade
import androidx.compose.animation.animateContentSize
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.res.colorResource
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.app.ActivityCompat
import androidx.core.app.NotificationCompat
import androidx.core.app.NotificationManagerCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import androidx.room.*
import androidx.work.*
import com.example.detector_spam.ui.theme.Detector_SpamTheme
import com.google.gson.Gson
import com.google.gson.GsonBuilder
import com.google.gson.annotations.SerializedName
import com.google.gson.reflect.TypeToken
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import java.util.concurrent.TimeUnit

// ==========================================
// 1. MODELOS DE DATOS Y ROOM DATABASE
// ==========================================
data class SmsModel(
    @SerializedName("id") val id: String,
    @SerializedName("remitente") val remitente: String,
    @SerializedName("mensaje") val mensaje: String
)

data class GrupoSpam(
    @SerializedName("remitente") val remitente: String,
    @SerializedName("resumenIa", alternate = ["resumenIA", "resumen_ia"]) val resumenIa: String,
    @SerializedName("mensajesOriginales", alternate = ["mensajes_originales"]) val mensajesOriginales: List<SmsModel>
)

data class EstadoApi(
    val mostrar: Boolean = false,
    val esExito: Boolean = false,
    val mensaje: String = ""
)

@Entity(tableName = "tabla_spam")
data class GrupoSpamEntity(
    @PrimaryKey val remitente: String,
    val resumenIa: String,
    val mensajesJson: String
)

@Dao
interface SpamDao {
    @Query("SELECT * FROM tabla_spam")
    fun obtenerTodoElSpamEnTiempoReal(): Flow<List<GrupoSpamEntity>>

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertarSpam(spam: List<GrupoSpamEntity>)

    @Query("DELETE FROM tabla_spam")
    suspend fun borrarTodo()
}

@Database(entities = [GrupoSpamEntity::class], version = 1, exportSchema = false)
abstract class AppDatabase : RoomDatabase() {
    abstract fun spamDao(): SpamDao
}

object DatabaseProvider {
    @Volatile
    private var INSTANCE: AppDatabase? = null

    fun getDatabase(context: Context): AppDatabase {
        return INSTANCE ?: synchronized(this) {
            val instance = Room.databaseBuilder(
                context.applicationContext,
                AppDatabase::class.java,
                "spam-database"
            ).build()
            INSTANCE = instance
            instance
        }
    }
}

enum class RutasPantalla(val titulo: String, val icono: ImageVector) {
    SPAM("Spam", Icons.Default.Warning),
    MENSAJES("Mensajes", Icons.Default.Email),
    AJUSTES("Ajustes", Icons.Default.Settings)
}

// ==========================================
// 2. ACTIVIDAD PRINCIPAL
// ==========================================
class MainActivity : ComponentActivity() {

    private var listaTodosLosMensajes = mutableStateListOf<SmsModel>()
    private var listaSpamAgrupado = mutableStateListOf<GrupoSpam>()
    private var isLoading = mutableStateOf(false)
    private var estadoApi = mutableStateOf(EstadoApi())
    private lateinit var db: AppDatabase

    private var esPrimeraVez = mutableStateOf(false)
    private lateinit var sharedPrefs: SharedPreferences

    private var smsPendientePorAnalizar = mutableStateOf<SmsModel?>(null)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        sharedPrefs = getSharedPreferences("AjustesSpamApp", Context.MODE_PRIVATE)
        esPrimeraVez.value = sharedPrefs.getBoolean("es_primera_vez", true)

        db = DatabaseProvider.getDatabase(applicationContext)

        crearCanalNotificaciones()
        pedirPermisoNotificacion()
        revisarIntent(intent)

        addOnNewIntentListener { nuevoIntent ->
            revisarIntent(nuevoIntent)
        }

        enableEdgeToEdge(
            statusBarStyle = SystemBarStyle.light(
                android.graphics.Color.TRANSPARENT,
                android.graphics.Color.TRANSPARENT
            ),
            navigationBarStyle = SystemBarStyle.light(
                android.graphics.Color.TRANSPARENT,
                android.graphics.Color.TRANSPARENT
            )
        )

        setContent {
            Detector_SpamTheme {
                if (esPrimeraVez.value) {
                    PantallaBienvenida(onEmpezarClick = {
                        sharedPrefs.edit().putBoolean("es_primera_vez", false).apply()
                        esPrimeraVez.value = false
                        observarBaseDeDatos()
                    })
                } else {
                    AppPrincipal(
                        todosLosMensajes = listaTodosLosMensajes,
                        spamAgrupado = listaSpamAgrupado,
                        isLoading = isLoading.value,
                        estadoApi = estadoApi.value,
                        onCerrarAlertaApi = { estadoApi.value = estadoApi.value.copy(mostrar = false) },
                        onActualizarClick = { checkearPermisosYExtraer() },
                        onExportarClick = { exportarJsonAlPortapapeles() },
                        onSimularSms = { simularLlegadaSms() },
                        onChecarLinea = { checador_linea() },
                        onProbarWorkManager = { simularWorkManager() },
                        onEncenderEscudo = { encenderEscudoInmortal() }
                    )
                }

                // DIÁLOGO EMERGENTE DE NUEVO SMS (Contraste Corregido)
                smsPendientePorAnalizar.value?.let { sms ->
                    AlertDialog(
                        onDismissRequest = { smsPendientePorAnalizar.value = null },
                        title = {
                            Text("¿Analizar nuevo mensaje?", fontWeight = FontWeight.Bold, color = colorResource(id = R.color.blue_dark))
                        },
                        text = {
                            // CORRECCIÓN: Se asigna Color.DarkGray para que siempre sea legible
                            Text("Recibiste un mensaje de:\n${sms.remitente}\n\n¿Deseas que sea analizado para detectar si es SPAM o no?", color = Color.DarkGray)
                        },
                        confirmButton = {
                            Button(
                                onClick = {
                                    procesarSmsIndividual(sms.remitente, sms.mensaje)
                                    smsPendientePorAnalizar.value = null
                                },
                                colors = ButtonDefaults.buttonColors(containerColor = colorResource(id = R.color.blue_primary))
                            ) {
                                Text("Sí, analizar", color = Color.White)
                            }
                        },
                        dismissButton = {
                            TextButton(onClick = { smsPendientePorAnalizar.value = null }) {
                                Text("Ignorar", color = Color.Gray)
                            }
                        },
                        containerColor = Color.White,
                        shape = RoundedCornerShape(16.dp)
                    )
                }
            }
        }

        if (!esPrimeraVez.value) {
            observarBaseDeDatos()
        }
    }

    private fun revisarIntent(intent: Intent) {
        if (intent.getBooleanExtra("preguntar_analisis", false)) {
            val remitente = intent.getStringExtra("nuevo_sms_remitente") ?: "Desconocido"
            val cuerpo = intent.getStringExtra("nuevo_sms_cuerpo") ?: ""
            smsPendientePorAnalizar.value = SmsModel("0", remitente, cuerpo)
        }
    }

    private fun crearCanalNotificaciones() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val name = "Alertas de Spam"
            val descriptionText = "Notificaciones cuando llega un mensaje sospechoso"
            val importance = NotificationManager.IMPORTANCE_HIGH
            val channel = NotificationChannel("CANAL_SPAM", name, importance).apply {
                description = descriptionText
            }
            val notificationManager: NotificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            notificationManager.createNotificationChannel(channel)
        }
    }

    private fun pedirPermisoNotificacion() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.POST_NOTIFICATIONS), 102)
        }
    }

    private fun observarBaseDeDatos() {
        lifecycleScope.launch(Dispatchers.IO) {
            db.spamDao().obtenerTodoElSpamEnTiempoReal().collect { spamGuardado ->
                val gson = Gson()
                val tipoListaSms = object : TypeToken<List<SmsModel>>() {}.type
                val gruposReconstruidos = spamGuardado.map { entidad ->
                    val mensajes: List<SmsModel> = gson.fromJson(entidad.mensajesJson, tipoListaSms)
                    GrupoSpam(entidad.remitente, entidad.resumenIa, mensajes)
                }
                withContext(Dispatchers.Main) {
                    listaSpamAgrupado.clear()
                    listaSpamAgrupado.addAll(gruposReconstruidos)
                }
            }
        }
    }

    private fun checkearPermisosYExtraer() {
        val permisoLeer = ContextCompat.checkSelfPermission(this, Manifest.permission.READ_SMS)
        val permisoRecibir = ContextCompat.checkSelfPermission(this, Manifest.permission.RECEIVE_SMS)

        if (permisoLeer == PackageManager.PERMISSION_GRANTED && permisoRecibir == PackageManager.PERMISSION_GRANTED) {
            extraerMensajesGuardados()
            if (listaTodosLosMensajes.isNotEmpty()) {
                enviarMensajesAApi(listaTodosLosMensajes.toList())
            }
        } else {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.READ_SMS, Manifest.permission.RECEIVE_SMS), 101)
        }
    }

    private fun encenderEscudoInmortal() {
        val intentService = Intent(this, EscudoService::class.java)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            startForegroundService(intentService)
        } else {
            startService(intentService)
        }
        Toast.makeText(this, "Escudo Inmortal Activado", Toast.LENGTH_SHORT).show()
    }

    private fun simularLlegadaSms() {
        procesarSmsIndividual("5512345678", "¡URGENTE! Tu cuenta ha sido bloqueada. Entra aqui: bit.ly/falso")
    }

    private fun simularWorkManager() {
        val remitente = "999999999"
        val cuerpo = "Este es un mensaje procesado por WorkManager en segundo plano."
        Toast.makeText(this, "Enviando tarea al WorkManager...", Toast.LENGTH_SHORT).show()

        val datos = workDataOf("remitente" to remitente, "cuerpo" to cuerpo)
        val restricciones = Constraints.Builder().setRequiredNetworkType(NetworkType.CONNECTED).build()
        val peticion = OneTimeWorkRequestBuilder<SpamCheckWorker>()
            .setInputData(datos)
            .setConstraints(restricciones)
            .build()
        WorkManager.getInstance(applicationContext).enqueue(peticion)
    }

    private fun procesarSmsIndividual(remitente: String, cuerpo: String) {
        mostrarNotificacionAnalisis()
        val mensajeUnico = listOf(SmsModel("0", remitente, cuerpo))
        enviarMensajesAApi(mensajeUnico)
    }

    private fun mostrarNotificacionAnalisis() {
        val builder = NotificationCompat.Builder(this, "CANAL_SPAM")
            .setSmallIcon(android.R.drawable.stat_notify_chat)
            .setContentTitle("Analizando mensaje...")
            .setContentText("Enviando datos a la IA para su revisión.")
            .setPriority(NotificationCompat.PRIORITY_HIGH)

        with(NotificationManagerCompat.from(this)) {
            if (ActivityCompat.checkSelfPermission(this@MainActivity, Manifest.permission.POST_NOTIFICATIONS) == PackageManager.PERMISSION_GRANTED) {
                notify(1, builder.build())
            }
        }
    }

    private fun extraerMensajesGuardados() {
        listaTodosLosMensajes.clear()
        val uri = Uri.parse("content://sms/inbox")
        val cursor = contentResolver.query(uri, null, null, null, null)

        cursor?.use {
            val indexId = it.getColumnIndex("_id")
            val indexAddress = it.getColumnIndex("address")
            val indexBody = it.getColumnIndex("body")

            var count = 0
            while (it.moveToNext() && count < 500) {
                val id = if (indexId != -1) it.getString(indexId) ?: "0" else "0"
                val sender = if (indexAddress != -1) it.getString(indexAddress) ?: "Desconocido" else "Desconocido"
                val body = if (indexBody != -1) it.getString(indexBody) ?: "" else ""

                listaTodosLosMensajes.add(SmsModel(id, sender, body))
                count++
            }
        }
    }

    private fun exportarJsonAlPortapapeles() {
        if (listaTodosLosMensajes.isEmpty()) checkearPermisosYExtraer()
        val gson = GsonBuilder().setPrettyPrinting().create()
        val jsonString = gson.toJson(listaTodosLosMensajes)

        try {
            val clipboard = getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
            val clip = ClipData.newPlainText("SMS JSON", jsonString)
            clipboard.setPrimaryClip(clip)
            Toast.makeText(this, "¡JSON copiado! (${listaTodosLosMensajes.size} mensajes)", Toast.LENGTH_LONG).show()
        } catch (e: Exception) {
            Toast.makeText(this, "Error al copiar: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }

    private fun enviarMensajesAApi(mensajesExtraidos: List<SmsModel>) {
        isLoading.value = true
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val mensajesAEnviar = mensajesExtraidos.toMutableList()
                mensajesAEnviar.add(SmsModel("999", "Aviso_Paqueteria", "Tu paquete de MercadoLibre no pudo ser entregado por direccion incorrecta. Actualiza tus datos de envio aqui o sera devuelto: http://correos-mexico-rastreo-fake.com/auth"))

                val gson = Gson()
                val jsonBody = gson.toJson(mensajesAEnviar)
                val requestBody = jsonBody.toRequestBody("application/json".toMediaTypeOrNull())

                val request = Request.Builder()
                    .url("http://192.168.50.130:8000/api/sms/clasificar-sms")
                    .post(requestBody)
                    .build()

                val client = OkHttpClient.Builder().connectTimeout(60, TimeUnit.SECONDS).writeTimeout(60, TimeUnit.SECONDS).readTimeout(60, TimeUnit.SECONDS).build()
                val response = client.newCall(request).execute()

                if (response.isSuccessful) {
                    val responseBody = response.body?.string()
                    val tipoLista = object : TypeToken<List<GrupoSpam>>() {}.type
                    val gruposSpamDevueltos: List<GrupoSpam> = gson.fromJson(responseBody, tipoLista) ?: emptyList()

                    if (gruposSpamDevueltos.isNotEmpty()) {
                        val entidades = gruposSpamDevueltos.map { GrupoSpamEntity(it.remitente, it.resumenIa, gson.toJson(it.mensajesOriginales)) }
                        db.spamDao().borrarTodo()
                        db.spamDao().insertarSpam(entidades)
                    }

                    withContext(Dispatchers.Main) {
                        isLoading.value = false
                        Toast.makeText(this@MainActivity, "¡Listo! ${gruposSpamDevueltos.size} amenazas detectadas.", Toast.LENGTH_SHORT).show()
                    }
                } else {
                    withContext(Dispatchers.Main) {
                        isLoading.value = false
                        Toast.makeText(this@MainActivity, "Error de servidor: ${response.code}", Toast.LENGTH_LONG).show()
                    }
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    isLoading.value = false
                    Toast.makeText(this@MainActivity, "Fallo de conexión. Posible servidor caido.", Toast.LENGTH_LONG).show()
                }
            }
        }
    }

    private fun checador_linea() {
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val client = OkHttpClient.Builder().connectTimeout(5, TimeUnit.SECONDS).readTimeout(5, TimeUnit.SECONDS).build()
                val request = Request.Builder().url("http://192.168.50.130:8000").get().build()
                val response = client.newCall(request).execute()

                if (response.isSuccessful) {
                    val bodyString = response.body?.string() ?: "Conectado correctamente"
                    withContext(Dispatchers.Main) { estadoApi.value = EstadoApi(mostrar = true, esExito = true, mensaje = bodyString) }
                } else {
                    withContext(Dispatchers.Main) { estadoApi.value = EstadoApi(mostrar = true, esExito = false, mensaje = "Error HTTP: ${response.code}") }
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) { estadoApi.value = EstadoApi(mostrar = true, esExito = false, mensaje = "Fallo en la conexión.") }
            }
        }
    }
}

// ==========================================
// 3. COMPOSABLES DE INTERFAZ (UI)
// ==========================================

@Composable
fun PantallaBienvenida(onEmpezarClick: () -> Unit) {
    Column(
        modifier = Modifier.fillMaxSize().background(Color.White).padding(24.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Icon(Icons.Default.Security, contentDescription = null, tint = colorResource(id = R.color.blue_primary), modifier = Modifier.size(100.dp))
        Spacer(modifier = Modifier.height(32.dp))
        Text("Bienvenido a Detector de Spam SMS", fontSize = 24.sp, fontWeight = FontWeight.Bold, color = colorResource(id = R.color.blue_dark), textAlign = TextAlign.Center)
        Spacer(modifier = Modifier.height(16.dp))
        Text("Identifica automáticamente contenido sospechoso, manteniendo tu bandeja limpia y segura.", fontSize = 16.sp, color = Color.Gray, textAlign = TextAlign.Center)
        Spacer(modifier = Modifier.height(48.dp))
        Button(
            onClick = onEmpezarClick,
            colors = ButtonDefaults.buttonColors(containerColor = colorResource(id = R.color.blue_primary)),
            shape = RoundedCornerShape(12.dp),
            modifier = Modifier.fillMaxWidth().height(50.dp)
        ) {
            Text("Empezar", fontSize = 16.sp, fontWeight = FontWeight.Bold)
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun AppPrincipal(
    todosLosMensajes: List<SmsModel>,
    spamAgrupado: List<GrupoSpam>,
    isLoading: Boolean,
    estadoApi: EstadoApi,
    onCerrarAlertaApi: () -> Unit,
    onActualizarClick: () -> Unit,
    onExportarClick: () -> Unit,
    onSimularSms: () -> Unit,
    onChecarLinea: () -> Unit,
    onProbarWorkManager: () -> Unit,
    onEncenderEscudo: () -> Unit
) {
    var pantallaActual by remember { mutableStateOf(RutasPantalla.SPAM) }

    // STATE HOISTING: Controlamos el modo global desde aquí
    var modoDesarrollador by remember { mutableStateOf(false) }
    var mostrandoTransicion by remember { mutableStateOf(false) }

    // COLORES DINÁMICOS
    val colorAcentoActivo = if (modoDesarrollador) Color(0xFFC62828) else colorResource(id = R.color.blue_primary)
    val colorFondoGeneral = if (modoDesarrollador) Color(0xFFFFF0F2) else colorResource(id = R.color.blue_light).copy(alpha = 0.5f)

    Box(modifier = Modifier.fillMaxSize()) {
        Scaffold(
            topBar = {
                TopAppBar(
                    title = {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Icon(if (modoDesarrollador) Icons.Default.Android else Icons.Default.Lock, contentDescription = null, tint = colorAcentoActivo, modifier = Modifier.size(24.dp))
                            Spacer(modifier = Modifier.width(8.dp))
                            Text(if (modoDesarrollador) "Desarrollador de SPAM" else "Detector Spam", color = colorAcentoActivo, fontWeight = FontWeight.SemiBold, fontSize = 20.sp)
                        }
                    },
                    colors = TopAppBarDefaults.topAppBarColors(containerColor = Color.White),
                    actions = {
                        if (pantallaActual != RutasPantalla.AJUSTES) {
                            TextButton(onClick = onActualizarClick, enabled = !isLoading) {
                                Text("Actualizar", color = if(isLoading) Color.Gray else colorAcentoActivo, fontSize = 14.sp)
                            }
                        }
                    }
                )
            },
            containerColor = colorFondoGeneral, // Fondo camaleónico
            bottomBar = {
                NavigationBar(containerColor = Color.White) {
                    RutasPantalla.values().forEach { pantalla ->
                        NavigationBarItem(
                            icon = { Icon(pantalla.icono, contentDescription = null) },
                            label = { Text(pantalla.titulo, fontSize = 12.sp) },
                            selected = pantallaActual == pantalla,
                            onClick = { if (!isLoading) pantallaActual = pantalla },
                            colors = NavigationBarItemDefaults.colors(
                                selectedIconColor = colorAcentoActivo, // Ícono dinámico
                                selectedTextColor = colorAcentoActivo, // Texto dinámico
                                unselectedIconColor = Color.Gray,
                                unselectedTextColor = Color.Gray,
                                indicatorColor = Color.Transparent
                            )
                        )
                    }
                }
            }
        ) { innerPadding ->
            Box(modifier = Modifier.padding(innerPadding).fillMaxSize()) {
                when (pantallaActual) {
                    RutasPantalla.SPAM -> PantallaSpam(spamAgrupado, onActualizarClick, colorAcentoActivo)
                    RutasPantalla.MENSAJES -> PantallaMensajes(todosLosMensajes, onActualizarClick, colorAcentoActivo)
                    RutasPantalla.AJUSTES -> PantallaAjustes(
                        modoDesarrollador = modoDesarrollador,
                        onModoDesarrolladorChange = { nuevoEstado ->
                            mostrandoTransicion = true
                            // Simulamos la carga del entorno
                            kotlin.concurrent.thread {
                                Thread.sleep(1200)
                                modoDesarrollador = nuevoEstado
                                mostrandoTransicion = false
                            }
                        },
                        onExportarClick = onExportarClick,
                        onSimularSms = onSimularSms,
                        onChecarLinea = onChecarLinea,
                        onProbarWorkManager = onProbarWorkManager,
                        onEncenderEscudo = onEncenderEscudo
                    )
                }
            }
        }

        if (isLoading) { PantallaCargando(modoDesarrollador) }

        if (mostrandoTransicion) { PantallaTransicionHacker(modoDesarrollador) }

        if (estadoApi.mostrar) {
            val colorFondo = if (estadoApi.esExito) Color(0xFFE8F5E9) else Color(0xFFFFEBEE)
            val colorTexto = if (estadoApi.esExito) Color(0xFF2E7D32) else Color(0xFFC62828)
            val titulo = if (estadoApi.esExito) "Conexión Exitosa" else "Fallo de Conexión"
            val icono = if (estadoApi.esExito) Icons.Default.CheckCircle else Icons.Default.Error

            AlertDialog(
                onDismissRequest = onCerrarAlertaApi,
                title = {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(icono, contentDescription = null, tint = colorTexto, modifier = Modifier.size(28.dp))
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(text = titulo, color = colorTexto, fontWeight = FontWeight.Bold, fontSize = 20.sp)
                    }
                },
                text = { Text(text = estadoApi.mensaje, color = Color.DarkGray, fontSize = 16.sp) }, // Contraste corregido
                confirmButton = { TextButton(onClick = onCerrarAlertaApi) { Text("Aceptar", color = colorTexto, fontWeight = FontWeight.Bold) } },
                containerColor = colorFondo, shape = RoundedCornerShape(16.dp)
            )
        }
    }
}

@Composable
fun PantallaTransicionHacker(haciaRedTeam: Boolean) {
    val colorProgreso = if (!haciaRedTeam) Color(0xFFC62828) else colorResource(id = R.color.blue_primary)
    val mensaje = if (!haciaRedTeam) "Iniciando Entorno de dsarrollador...\nHackeando Android" else "Regresando con los mortales \nVolviendo a Detector de Spam"

    Box(modifier = Modifier.fillMaxSize().background(Color.White.copy(alpha = 0.98f)).clickable(enabled = false) {}, contentAlignment = Alignment.Center) {
        Column(horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.Center) {
            CircularProgressIndicator(color = colorProgreso, strokeWidth = 5.dp, modifier = Modifier.size(64.dp))
            Spacer(modifier = Modifier.height(32.dp))
            Text(text = mensaje, fontSize = 16.sp, fontWeight = FontWeight.Bold, color = Color.DarkGray, textAlign = TextAlign.Center)
        }
    }
}

@Composable
fun PantallaCargando(esRedTeam: Boolean) {
    val colorProgreso = if (esRedTeam) Color(0xFFC62828) else colorResource(id = R.color.blue_primary)
    val colorTexto = if (esRedTeam) Color(0xFFC62828) else colorResource(id = R.color.blue_dark)

    val mensajes = listOf("Cargando tus mensajes...", "Detectando spam...", "La IA te dará un resumen", "Has pensado en cuantos mensajes tienes?", "Preparate para usar mucho esta app...")
    var mensajeActualIndex by remember { mutableStateOf(0) }
    LaunchedEffect(Unit) {
        while (true) {
            delay(2500)
            mensajeActualIndex = (mensajeActualIndex + 1) % mensajes.size
        }
    }
    Box(modifier = Modifier.fillMaxSize().background(Color.White.copy(alpha = 0.95f)).clickable(enabled = false) {}, contentAlignment = Alignment.Center) {
        Column(horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.Center) {
            CircularProgressIndicator(color = colorProgreso, strokeWidth = 5.dp, modifier = Modifier.size(64.dp))
            Spacer(modifier = Modifier.height(32.dp))
            Text(text = mensajes[mensajeActualIndex], fontSize = 18.sp, fontWeight = FontWeight.Medium, color = colorTexto, textAlign = TextAlign.Center, modifier = Modifier.padding(horizontal = 32.dp).animateContentSize(animationSpec = tween(500)))
        }
    }
}

@Composable
fun PantallaSpam(spamAgrupado: List<GrupoSpam>, onActualizarClick: () -> Unit, colorAcento: Color) {
    if (spamAgrupado.isEmpty()) {
        PantallaVaciaConBoton(mensaje = "NO HAY SPAM DETECTADO", onActualizarClick = onActualizarClick, colorAcento)
    } else {
        LazyColumn(modifier = Modifier.fillMaxSize().padding(horizontal = 12.dp, vertical = 8.dp)) {
            items(spamAgrupado) { grupo -> TarjetaGrupoSpamExpansible(grupo, colorAcento) }
        }
    }
}

@Composable
fun PantallaMensajes(mensajes: List<SmsModel>, onActualizarClick: () -> Unit, colorAcento: Color) {
    if (mensajes.isEmpty()) {
        PantallaVaciaConBoton(mensaje = "NO HAY MENSAJES CARGADOS", onActualizarClick = onActualizarClick, colorAcento)
    } else {
        LazyColumn(modifier = Modifier.fillMaxSize().padding(horizontal = 12.dp, vertical = 8.dp)) {
            items(mensajes) { sms -> TarjetaSmsMinimalista(sms) }
        }
    }
}

@Composable
fun PantallaVaciaConBoton(mensaje: String, onActualizarClick: () -> Unit, colorAcento: Color) {
    Column(modifier = Modifier.fillMaxSize(), verticalArrangement = Arrangement.Center, horizontalAlignment = Alignment.CenterHorizontally) {
        Icon(Icons.Default.Info, contentDescription = null, tint = Color.Gray.copy(alpha = 0.6f), modifier = Modifier.size(64.dp))
        Spacer(modifier = Modifier.height(16.dp))
        Text(text = mensaje, color = colorAcento, fontWeight = FontWeight.Bold, fontSize = 18.sp, textAlign = TextAlign.Center)
        Spacer(modifier = Modifier.height(24.dp))
        Button(onClick = onActualizarClick, colors = ButtonDefaults.buttonColors(containerColor = colorAcento), shape = RoundedCornerShape(12.dp)) {
            Icon(Icons.Default.Refresh, contentDescription = null, modifier = Modifier.size(18.dp))
            Spacer(modifier = Modifier.width(8.dp))
            Text("Actualizar", fontSize = 14.sp)
        }
    }
}

@Composable
fun TarjetaSmsMinimalista(sms: SmsModel) {
    Card(modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp), colors = CardDefaults.cardColors(containerColor = Color.White), elevation = CardDefaults.cardElevation(defaultElevation = 2.dp), shape = RoundedCornerShape(12.dp)) {
        Row(modifier = Modifier.padding(12.dp), verticalAlignment = Alignment.CenterVertically) {
            Icon(Icons.Default.Email, contentDescription = null, tint = Color.Gray.copy(alpha = 0.6f), modifier = Modifier.size(24.dp))
            Spacer(modifier = Modifier.width(12.dp))
            Column {
                Text(text = sms.remitente, fontWeight = FontWeight.Bold, fontSize = 14.sp, color = Color.DarkGray)
                Spacer(modifier = Modifier.height(4.dp))
                Text(text = sms.mensaje, fontSize = 13.sp, color = Color.Gray, maxLines = 3)
            }
        }
    }
}

@Composable
fun TarjetaGrupoSpamExpansible(grupo: GrupoSpam, colorAcento: Color) {
    var expanded by remember { mutableStateOf(false) }
    Card(
        modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp).animateContentSize(),
        colors = CardDefaults.cardColors(containerColor = Color.White), elevation = CardDefaults.cardElevation(defaultElevation = 2.dp), shape = RoundedCornerShape(12.dp)
    ) {
        Column(modifier = Modifier.clickable { expanded = !expanded }.padding(12.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(Icons.Default.Warning, contentDescription = null, tint = Color.Gray.copy(alpha = 0.6f), modifier = Modifier.size(24.dp))
                Spacer(modifier = Modifier.width(12.dp))
                Column(modifier = Modifier.weight(1f)) {
                    Text(text = grupo.remitente, fontWeight = FontWeight.Bold, fontSize = 14.sp, color = Color.DarkGray)
                    Spacer(modifier = Modifier.height(2.dp))
                    Text(text = "${grupo.mensajesOriginales.size} mensajes de spam", fontSize = 12.sp, color = Color.Gray)
                }
                Icon(if (expanded) Icons.Default.KeyboardArrowUp else Icons.Default.KeyboardArrowDown, contentDescription = null, tint = colorAcento)
            }
            Spacer(modifier = Modifier.height(10.dp))
            Text(text = grupo.resumenIa, fontSize = 13.sp, color = colorAcento, fontWeight = FontWeight.SemiBold)

            if (expanded) {
                Divider(modifier = Modifier.padding(vertical = 12.dp), color = Color.Gray.copy(alpha = 0.2f))
                Column(modifier = Modifier.fillMaxWidth().background(Color.Transparent)) {
                    val mensajesAMostrar = grupo.mensajesOriginales.take(5)
                    mensajesAMostrar.forEach { sms ->
                        Column(modifier = Modifier.padding(bottom = 12.dp)) {
                            Text(text = sms.mensaje, fontSize = 13.sp, color = Color.DarkGray)
                        }
                    }
                    if (grupo.mensajesOriginales.size > 5) {
                        Text(text = "... y ${grupo.mensajesOriginales.size - 5} mensajes más ocultos", fontSize = 12.sp, color = Color.Gray, fontStyle = FontStyle.Italic, modifier = Modifier.padding(top = 4.dp))
                    }
                }
            }
        }
    }
}

// ==========================================
// PANTALLA DE AJUSTES REFACTORIZADA
// ==========================================
@Composable
fun PantallaAjustes(
    modoDesarrollador: Boolean,
    onModoDesarrolladorChange: (Boolean) -> Unit,
    onExportarClick: () -> Unit,
    onSimularSms: () -> Unit,
    onChecarLinea: () -> Unit,
    onProbarWorkManager: () -> Unit,
    onEncenderEscudo: () -> Unit
) {
    var mostrarSoporte by remember { mutableStateOf(false) }

    Crossfade(targetState = modoDesarrollador, animationSpec = tween(400)) { esModoDev ->
        if (esModoDev) {
            // ===============================================
            // VISTA DESARROLLADOR
            // ===============================================
            val redPrimario = Color(0xFFC62828)

            Column(modifier = Modifier.fillMaxSize().padding(horizontal = 16.dp, vertical = 24.dp)) {
                Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.padding(bottom = 16.dp)) {
                    IconButton(onClick = { onModoDesarrolladorChange(false) }) {
                        Icon(Icons.Default.ArrowBack, contentDescription = "Volver", tint = redPrimario)
                    }
                    Text(text = "Modo Desarrollador", fontSize = 18.sp, fontWeight = FontWeight.Bold, color = redPrimario)
                }

                Card(
                    modifier = Modifier.fillMaxWidth().padding(bottom = 16.dp),
                    colors = CardDefaults.cardColors(containerColor = Color(0xFFFFEBEE))
                ) {
                    Text(
                        "Entorno de Pruebas. Utiliza estas herramientas para verficar el correcto funcionamiento de la app",
                        modifier = Modifier.padding(12.dp),
                        color = Color(0xFFB71C1C),
                        fontWeight = FontWeight.SemiBold,
                        fontSize = 13.sp
                    )
                }

                AjusteItem(icono = Icons.Default.ContentCopy, titulo = "Extraer SMS en json", subtitulo = "Copia los SMS al portapapeles", onClick = onExportarClick, colorIcono = redPrimario)
                Spacer(modifier = Modifier.height(12.dp))
                AjusteItem(icono = Icons.Default.NotificationsActive, titulo = "Inyectar Payload Local", subtitulo = "Simular notificación en tiempo real", onClick = onSimularSms, colorIcono = redPrimario)
                Spacer(modifier = Modifier.height(12.dp))
                AjusteItem(icono = Icons.Default.Engineering, titulo = "Forzar Worker", subtitulo = "Probar Worker en segundo plano", onClick = onProbarWorkManager, colorIcono = redPrimario)
                Spacer(modifier = Modifier.height(12.dp))
                AjusteItem(icono = Icons.Default.AddTask, titulo = "Verificar conexcion a la API", subtitulo = "Verificar estado del servidor", onClick = onChecarLinea, colorIcono = redPrimario)
                Spacer(modifier = Modifier.height(12.dp))
                AjusteItem(icono = Icons.Default.VerifiedUser, titulo = "Forzar Escudo Inmortal", subtitulo = "Activa la opción de aplicación despierta todo el tiempo", onClick = onEncenderEscudo, colorIcono = redPrimario)

                Spacer(modifier = Modifier.weight(1f))
                Text(text = "Entorno Dev V2.9", fontSize = 11.sp, color = Color.Gray, modifier = Modifier.fillMaxWidth(), textAlign = TextAlign.Center)
            }
        } else {
            Column(modifier = Modifier.fillMaxSize().padding(horizontal = 16.dp, vertical = 24.dp)) {
                Text(text = "Configuración", fontSize = 18.sp, fontWeight = FontWeight.Bold, color = colorResource(id = R.color.blue_dark), modifier = Modifier.padding(bottom = 16.dp, start = 4.dp))

                AjusteItem(icono = Icons.Default.SupportAgent, titulo = "Soporte Técnico", subtitulo = "Contáctanos para ayuda o sugerencias", onClick = { mostrarSoporte = true }, colorIcono = colorResource(id = R.color.blue_primary))
                Spacer(modifier = Modifier.height(12.dp))
                AjusteItem(icono = Icons.Default.Terminal, titulo = "Opciones de Desarrollador", subtitulo = "Herramientas avanzadas de testing", onClick = { onModoDesarrolladorChange(true) }, colorIcono = Color.Gray)

                Spacer(modifier = Modifier.weight(1f))
                Text(text = "Detector Spam V2.9", fontSize = 11.sp, color = Color.Gray, modifier = Modifier.fillMaxWidth(), textAlign = TextAlign.Center)
            }
        }
    }

    if (mostrarSoporte) {
        AlertDialog(
            onDismissRequest = { mostrarSoporte = false },
            title = {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(Icons.Default.SupportAgent, contentDescription = null, tint = colorResource(id = R.color.blue_primary))
                    Spacer(modifier = Modifier.width(8.dp))
                    // CORRECCIÓN: Título de Soporte con color oscuro explícito
                    Text("Soporte Técnico", fontWeight = FontWeight.Bold, color = colorResource(id = R.color.blue_dark))
                }
            },
            // CORRECCIÓN: Texto de Soporte con color oscuro explícito
            text = { Text("Si tienes problemas con la detección de Spam o necesitas reportar un fallo de red, contáctanos.\n\nCorreo: ayuda@detectorspam.com\n\n", color = Color.DarkGray) },
            confirmButton = { TextButton(onClick = { mostrarSoporte = false }) { Text("Cerrar", color = colorResource(id = R.color.blue_primary), fontWeight = FontWeight.Bold) } },
            containerColor = Color.White
        )
    }
}

@Composable
fun AjusteItem(icono: ImageVector, titulo: String, subtitulo: String, onClick: () -> Unit, colorIcono: Color) {
    Card(modifier = Modifier.fillMaxWidth().clickable { onClick() }, colors = CardDefaults.cardColors(containerColor = Color.White), elevation = CardDefaults.cardElevation(defaultElevation = 1.dp), shape = RoundedCornerShape(12.dp)) {
        Row(modifier = Modifier.padding(16.dp), verticalAlignment = Alignment.CenterVertically) {
            Icon(icono, contentDescription = null, tint = colorIcono, modifier = Modifier.size(28.dp))
            Spacer(modifier = Modifier.width(16.dp))
            Column(modifier = Modifier.weight(1f)) {
                Text(text = titulo, fontWeight = FontWeight.SemiBold, fontSize = 15.sp, color = Color.DarkGray) // Color explícito
                Spacer(modifier = Modifier.height(2.dp))
                Text(text = subtitulo, fontSize = 13.sp, color = Color.Gray)
            }
            Icon(Icons.Default.KeyboardArrowRight, contentDescription = null, tint = Color.LightGray, modifier = Modifier.size(24.dp))
        }
    }
}

// ==========================================
// 4. WORKMANAGER Y SERVICIO EN SEGUNDO PLANO
// ==========================================

class SpamCheckWorker(appContext: Context, workerParams: WorkerParameters) : CoroutineWorker(appContext, workerParams) {
    override suspend fun doWork(): Result = withContext(Dispatchers.IO) {
        val remitente = inputData.getString("remitente") ?: return@withContext Result.failure()
        val cuerpo = inputData.getString("cuerpo") ?: return@withContext Result.failure()

        return@withContext try {
            val client = OkHttpClient.Builder().connectTimeout(60, TimeUnit.SECONDS).readTimeout(60, TimeUnit.SECONDS).build()
            val gson = Gson()
            val smsJson = gson.toJson(listOf(SmsModel("0", remitente, cuerpo)))
            val body = smsJson.toRequestBody("application/json".toMediaTypeOrNull())
            val request = Request.Builder().url("http://192.168.50.130:8000/api/sms/clasificar-sms").post(body).build()

            val response = client.newCall(request).execute()

            if (response.isSuccessful) {
                val responseBody = response.body?.string()
                val tipoLista = object : TypeToken<List<GrupoSpam>>() {}.type
                val gruposSpamDevueltos: List<GrupoSpam> = gson.fromJson(responseBody, tipoLista) ?: emptyList()

                if (gruposSpamDevueltos.isNotEmpty()) {
                    val database = DatabaseProvider.getDatabase(applicationContext)
                    val entidades = gruposSpamDevueltos.map { GrupoSpamEntity(it.remitente, it.resumenIa, gson.toJson(it.mensajesOriginales)) }
                    database.spamDao().insertarSpam(entidades)
                    mostrarNotificacionSpamEncontrado()
                }
                Log.d("WorkManager", "¡Mensaje procesado en segundo plano con éxito!")
                Result.success()
            } else {
                Result.retry()
            }
        } catch (e: Exception) {
            Log.e("WorkManager", "Fallo al contactar API, reintentando... ${e.message}")
            Result.retry()
        }
    }

    private fun mostrarNotificacionSpamEncontrado() {
        val notificationManager = applicationContext.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel("CANAL_SPAM", "Alertas de Spam", NotificationManager.IMPORTANCE_HIGH)
            notificationManager.createNotificationChannel(channel)
        }
        val builder = NotificationCompat.Builder(applicationContext, "CANAL_SPAM")
            .setSmallIcon(android.R.drawable.stat_notify_chat)
            .setContentTitle("¡Escudo Activado! 🛡️")
            .setContentText("El WorkManager detectó y bloqueó una estafa.")
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            .setAutoCancel(true)

        if (ActivityCompat.checkSelfPermission(applicationContext, Manifest.permission.POST_NOTIFICATIONS) == PackageManager.PERMISSION_GRANTED) {
            notificationManager.notify(System.currentTimeMillis().toInt(), builder.build())
        }
    }
}