package com.bluetooth.mwoolley.microbitbledemo.ui;
/*
 * Author: Arjun Rao
 *
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

import android.content.ComponentName;
import android.content.Intent;
import android.content.ServiceConnection;
import android.os.Bundle;
import android.os.Handler;
import android.os.IBinder;
import android.os.Message;
import android.support.v7.app.AppCompatActivity;
import android.text.Html;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.Switch;
import android.widget.TextView;

import com.android.volley.AuthFailureError;
import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.Response;
import com.android.volley.VolleyError;
import com.android.volley.toolbox.StringRequest;
import com.android.volley.toolbox.Volley;
import com.bluetooth.mwoolley.microbitbledemo.Constants;
import com.bluetooth.mwoolley.microbitbledemo.MicroBit;
import com.bluetooth.mwoolley.microbitbledemo.R;
import com.bluetooth.mwoolley.microbitbledemo.Settings;
import com.bluetooth.mwoolley.microbitbledemo.Utility;
import com.bluetooth.mwoolley.microbitbledemo.bluetooth.BleAdapterService;
import com.bluetooth.mwoolley.microbitbledemo.bluetooth.ConnectionStatusListener;
import com.google.gson.Gson;

import org.json.JSONException;
import org.json.JSONObject;

import java.util.HashMap;
import java.util.Map;

public class AccelerometerActivity extends AppCompatActivity implements ConnectionStatusListener {

    private static final int ACCELEROMETER_G_RANGE = 2;
    private static final int ACCELEROMETER_DIVISOR = 512;
    public static final int SAMPLE_SIZE = 120;

    private float[] accel_input = new float[3];
    private float[] accel_output = new float[3];
    private Gson gson = new Gson();
    private double[][] record_data = new double[SAMPLE_SIZE][7];
    private boolean recording = false;
    private boolean is_prediction = false;
    private int next_rec_index = 0;

    private BleAdapterService bluetooth_le_adapter;
    private RequestQueue queue;

    private boolean exiting=false;
    private int accelerometer_period;

    private boolean notifications_on =false;
    private long start_time;
    private int minute_number;
    private int notification_count;
    private boolean apply_smoothing=true;

    private final ServiceConnection mServiceConnection = new ServiceConnection() {

        @Override
        public void onServiceConnected(ComponentName componentName, IBinder service) {
            Log.d(Constants.TAG,"onServiceConnected");
            notifications_on=false;
            start_time = System.currentTimeMillis();
            minute_number=1;
            notification_count=0;
            showBenchmark();
            bluetooth_le_adapter = ((BleAdapterService.LocalBinder) service).getService();
            bluetooth_le_adapter.setActivityHandler(mMessageHandler);
            bluetooth_le_adapter.readCharacteristic(Utility.normaliseUUID(BleAdapterService.ACCELEROMETERSERVICE_SERVICE_UUID),Utility.normaliseUUID(BleAdapterService.ACCELEROMETERPERIOD_CHARACTERISTIC_UUID));
            showMsg("Please wait... Receiving Data from MicroBit.");
        }

        @Override
        public void onServiceDisconnected(ComponentName componentName) {
        }

    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        this.getWindow().setSoftInputMode(WindowManager.LayoutParams.SOFT_INPUT_STATE_ALWAYS_HIDDEN);
        setContentView(R.layout.activity_accelerometer);
        getSupportActionBar().setTitle(R.string.screen_title_accelerometer);

        // read intent data
        final Intent intent = getIntent();
        MicroBit.getInstance().setConnection_status_listener(this);
        queue = Volley.newRequestQueue(this);

        // connect to the Bluetooth smart service
        Intent gattServiceIntent = new Intent(this, BleAdapterService.class);
        bindService(gattServiceIntent, mServiceConnection, BIND_AUTO_CREATE);
        showMsg("Please wait... Receiving Data from MicroBit.");
        toggleButtons(false);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (notifications_on) {
            bluetooth_le_adapter.setNotificationsState(Utility.normaliseUUID(BleAdapterService.ACCELEROMETERSERVICE_SERVICE_UUID), Utility.normaliseUUID(BleAdapterService.ACCELEROMETERDATA_CHARACTERISTIC_UUID), false);
        }
        try {
            // may already have unbound. No API to check state so....
            unbindService(mServiceConnection);
        } catch (Exception e) {
        }
    }

    public void onBackPressed() {
        Log.d(Constants.TAG, "onBackPressed");
        if (MicroBit.getInstance().isMicrobit_connected() && notifications_on) {
            bluetooth_le_adapter.setNotificationsState(Utility.normaliseUUID(BleAdapterService.ACCELEROMETERSERVICE_SERVICE_UUID), Utility.normaliseUUID(BleAdapterService.ACCELEROMETERDATA_CHARACTERISTIC_UUID), false);
        }
        exiting=true;
        if (!MicroBit.getInstance().isMicrobit_connected()) {
            try {
                // may already have unbound. No API to check state so....
                unbindService(mServiceConnection);
            } catch (Exception e) {
            }
            finish();
        }
        exiting=true;
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if it is present.
        getMenuInflater().inflate(R.menu.menu_accelerometer, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {

        int id = item.getItemId();

        if (id == R.id.menu_accelerometer_settings) {
            accelerometer_period = Settings.getInstance().getAccelerometer_period();
            Intent intent = new Intent(AccelerometerActivity.this, AccelerometerSettingsActivity.class);
            startActivityForResult(intent, AccelerometerSettingsActivity.START_ACCELEROMETER_SETTINGS);
            return true;
        }
        if (id == R.id.menu_accelerometer_help) {
            Intent intent = new Intent(AccelerometerActivity.this, HelpActivity.class);
            intent.putExtra(Constants.URI, Constants.ACCELEROMETER_HELP);
            startActivity(intent);
            return true;
        }
        return super.onOptionsItemSelected(item);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        Log.d(Constants.TAG, "onActivityResult");
        if (requestCode == AccelerometerSettingsActivity.START_ACCELEROMETER_SETTINGS) {
            if (resultCode == RESULT_OK) {
                Log.d(Constants.TAG, "onActivityResult RESULT_OK");
                if (Settings.getInstance().getAccelerometer_period() != accelerometer_period) {
                    Log.d(Constants.TAG, "accelerometer period has changed to "+Settings.getInstance().getAccelerometer_period());
                    accelerometer_period = Settings.getInstance().getAccelerometer_period();
                    showMsg(Utility.htmlColorBlue("Changing accelerometer period to:"));
                    Log.d(Constants.TAG, Utility.byteArrayAsHexString(Utility.leBytesFromShort(Settings.getInstance().getAccelerometer_period())));
                    bluetooth_le_adapter.writeCharacteristic(Utility.normaliseUUID(BleAdapterService.ACCELEROMETERSERVICE_SERVICE_UUID), Utility.normaliseUUID(BleAdapterService.ACCELEROMETERPERIOD_CHARACTERISTIC_UUID), Utility.leBytesFromShort(Settings.getInstance().getAccelerometer_period()));
                }
            } else {
                Log.d(Constants.TAG, "onActivityResult NOT RESULT_OK");
            }
        }
    }

    public void onApplySmoothingChanged(View v) {
        apply_smoothing = ((Switch) v).isChecked();
    }

    public void onRecordClick(View v) {
        if (notifications_on) {
            ((Button) v).setEnabled(false);
            next_rec_index = 0;
            is_prediction = false;
            recording = true;
        } else {
            showMsg("Still connecting to MicroBit...Please try again after MicroBit is Connected...");
        }
    }

    public void onPredictClick(View v) {
        if (notifications_on) {
            ((Button) v).setEnabled(false);
            next_rec_index = 0;
            recording = true;
            is_prediction = true;
        } else {
            showMsg("Still connecting to MicroBit...Please try again after MicroBit is Connected...");
        }
    }

    public void onTrainClick(View v) {
        if (notifications_on) {
            this.sendTrainRequest();
        } else {
            showMsg("Still connecting to MicroBit...Please try again after MicroBit is Connected...");
        }
    }


    private Handler mMessageHandler = new Handler() {
        @Override
        public void handleMessage(Message msg) {

            Bundle bundle;
            String service_uuid = "";
            String characteristic_uuid = "";
            String descriptor_uuid = "";
            byte[] b = null;
            TextView value_text = null;

            switch (msg.what) {
                case BleAdapterService.GATT_CHARACTERISTIC_READ:
                    Log.d(Constants.TAG, "Handler received characteristic read result");
                    bundle = msg.getData();
                    service_uuid = bundle.getString(BleAdapterService.PARCEL_SERVICE_UUID);
                    characteristic_uuid = bundle.getString(BleAdapterService.PARCEL_CHARACTERISTIC_UUID);
                    b = bundle.getByteArray(BleAdapterService.PARCEL_VALUE);
                    Log.d(Constants.TAG, "characteristic " + characteristic_uuid + " of service " + service_uuid + " read OK");
                    Log.d(Constants.TAG, "Value=" + Utility.byteArrayAsHexString(b));
                    if (characteristic_uuid.equalsIgnoreCase(Utility.normaliseUUID(BleAdapterService.ACCELEROMETERPERIOD_CHARACTERISTIC_UUID))) {
                        boolean got_accelerometer_period = false;
                        byte [] period_bytes = new byte[2];
                        if (b.length == 2) {
                            period_bytes[0] = b[0];
                            period_bytes[1] = b[1];
                            got_accelerometer_period = true;
                        } else {
                            if (b.length == 1) {
                                period_bytes[0] = b[0];
                                period_bytes[1] = 0x00;
                                got_accelerometer_period = true;
                            } else {
                                Log.d(Constants.TAG,"Couldn't obtain value of accelerometer period");
                            }
                        }
                        if (got_accelerometer_period) {
                            accelerometer_period = (int) Utility.shortFromLittleEndianBytes(period_bytes);
                            Settings.getInstance().setAccelerometer_period((short) accelerometer_period);
                            showAccelerometerPeriod();
                        }
                    }
                    bluetooth_le_adapter.setNotificationsState(Utility.normaliseUUID(BleAdapterService.ACCELEROMETERSERVICE_SERVICE_UUID), Utility.normaliseUUID(BleAdapterService.ACCELEROMETERDATA_CHARACTERISTIC_UUID), true);
                    break;
                case BleAdapterService.GATT_CHARACTERISTIC_WRITTEN:
                    Log.d(Constants.TAG, "Handler received characteristic written result");
                    bundle = msg.getData();
                    service_uuid = bundle.getString(BleAdapterService.PARCEL_SERVICE_UUID);
                    characteristic_uuid = bundle.getString(BleAdapterService.PARCEL_CHARACTERISTIC_UUID);
                    Log.d(Constants.TAG, "characteristic " + characteristic_uuid + " of service " + service_uuid + " written OK");
                    showAccelerometerPeriod();
                    showMsg(Utility.htmlColorGreen("Ready"));
                    break;
                case BleAdapterService.GATT_DESCRIPTOR_WRITTEN:
                    Log.d(Constants.TAG, "Handler received descriptor written result");
                    bundle = msg.getData();
                    service_uuid = bundle.getString(BleAdapterService.PARCEL_SERVICE_UUID);
                    characteristic_uuid = bundle.getString(BleAdapterService.PARCEL_CHARACTERISTIC_UUID);
                    descriptor_uuid = bundle.getString(BleAdapterService.PARCEL_DESCRIPTOR_UUID);
                    Log.d(Constants.TAG, "descriptor " + descriptor_uuid + " of characteristic " + characteristic_uuid + " of service " + service_uuid + " written OK");
                    if (!exiting) {
                        showMsg(Utility.htmlColorGreen("MicroBit Ready"));
                        notifications_on=true;
                        start_time = System.currentTimeMillis();
                        toggleButtons(true);
                    } else {
                        showMsg(Utility.htmlColorGreen("MicroBit Disconnected"));
                        notifications_on=false;
                        toggleButtons(false);
                        finish();
                    }
                    break;

                case BleAdapterService.NOTIFICATION_OR_INDICATION_RECEIVED:
                    bundle = msg.getData();
                    service_uuid = bundle.getString(BleAdapterService.PARCEL_SERVICE_UUID);
                    characteristic_uuid = bundle.getString(BleAdapterService.PARCEL_CHARACTERISTIC_UUID);
                    b = bundle.getByteArray(BleAdapterService.PARCEL_VALUE);
                    Log.d(Constants.TAG, "Value=" + Utility.byteArrayAsHexString(b));
                    if (characteristic_uuid.equalsIgnoreCase((Utility.normaliseUUID(BleAdapterService.ACCELEROMETERDATA_CHARACTERISTIC_UUID)))) {
                        notification_count++;
                        if (System.currentTimeMillis() - start_time >= 60000) {
                            showBenchmark();
                            notification_count = 0;
                            minute_number++;
                            start_time = System.currentTimeMillis();
                        }
                        byte[] x_bytes = new byte[2];
                        byte[] y_bytes = new byte[2];
                        byte[] z_bytes = new byte[2];
                        byte[] xb_bytes = new byte[2];
                        byte[] yb_bytes = new byte[2];
                        byte[] zb_bytes = new byte[2];
                        System.arraycopy(b, 0, x_bytes, 0, 2);
                        System.arraycopy(b, 2, y_bytes, 0, 2);
                        System.arraycopy(b, 4, z_bytes, 0, 2);
                        System.arraycopy(b, 6, xb_bytes, 0, 2);
                        System.arraycopy(b, 8, yb_bytes, 0, 2);
                        System.arraycopy(b, 10, zb_bytes, 0, 2);
                        short raw_x = Utility.shortFromLittleEndianBytes(x_bytes);
                        short raw_y = Utility.shortFromLittleEndianBytes(y_bytes);
                        short raw_z = Utility.shortFromLittleEndianBytes(z_bytes);
                        short raw_xb = Utility.shortFromLittleEndianBytes(xb_bytes); // Flex Sensor
                        short raw_yb = Utility.shortFromLittleEndianBytes(yb_bytes); // Force Sensor
                        short raw_zb = Utility.shortFromLittleEndianBytes(zb_bytes); // Accelerometer Period
                        Log.d(Constants.TAG, "Accelerometer Data received: x=" + raw_x + " y=" + raw_y + " z=" + raw_z);
                        Log.d(Constants.TAG, "Accelerometer B Data received: x=" + raw_xb + " y=" + raw_yb + " z=" + raw_zb);


                        // range is -1024 : +1024
                        // Starting with the LED display face up and level (perpendicular to gravity) and edge connector towards your body:
                        // A negative X value means tilting left, a positive X value means tilting right
                        // A negative Y value means tilting away from you, a positive Y value means tilting towards you
                        // A negative Z value means ?

                        accel_input[0] = raw_x / 1000f;
                        accel_input[1] = raw_y / 1000f;
                        accel_input[2] = raw_z / 1000f;
                        if (apply_smoothing) {
                            accel_output = Utility.lowPass(accel_input, accel_output);
                        } else {
                            accel_output[0] = accel_input[0];
                            accel_output[1] = accel_input[1];
                            accel_output[2] = accel_input[2];
                        }

                        double pitch = Math.atan(accel_output[0] / Math.sqrt(Math.pow(accel_output[1], 2) + Math.pow(accel_output[2], 2)));
                        double roll = Math.atan(accel_output[1] / Math.sqrt(Math.pow(accel_output[0], 2) + Math.pow(accel_output[2], 2)));
                        //convert radians into degrees
                        pitch = pitch * (180.0 / Math.PI);
                        roll = -1 * roll * (180.0 / Math.PI);
                        accelerometer_period = raw_zb;
                        Settings.getInstance().setAccelerometer_period((short) accelerometer_period);
                        showAccelerometerPeriod();
                        showAccelerometerData(accel_output,pitch,roll, raw_yb, raw_xb);

                    }
                    break;
                case BleAdapterService.GATT_REMOTE_RSSI:
                    bundle = msg.getData();
                    int rssi = bundle.getInt(BleAdapterService.PARCEL_RSSI);
//                    PeripheralControlActivity.this.updateRssi(rssi);
                    break;
                case BleAdapterService.MESSAGE:
                    bundle = msg.getData();
                    String text = bundle.getString(BleAdapterService.PARCEL_TEXT);
                    showMsg(Utility.htmlColorRed(text));
            }
        }
    };

    private void showMsg(final String msg) {
        Log.d(Constants.TAG, msg);
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                ((TextView) AccelerometerActivity.this.findViewById(R.id.message)).setText(Html.fromHtml(msg));
            }
        });
    }

    private void showAccelerometerPeriod() {
        Log.d(Constants.TAG, "Accelerometer Period: "+accelerometer_period+"ms");
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                ((TextView) AccelerometerActivity.this.findViewById(R.id.accel_period)).setText("Polling: "+Integer.toString(accelerometer_period)+"ms");
            }
        });
    }

    private void toggleButtons(boolean state) {
        ((Button) AccelerometerActivity.this.findViewById(R.id.record_btn)).setEnabled(state);
        ((Button) AccelerometerActivity.this.findViewById(R.id.train_btn)).setEnabled(state);
        ((Button) AccelerometerActivity.this.findViewById(R.id.predict_btn)).setEnabled(state);
    }

    private void showAccelerometerData(final float [] accel_data, final double pitch, final double roll, final double force, final double flex) {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                ((TextView) AccelerometerActivity.this.findViewById(R.id.accel_x)).setText("X: " + String.format("%.3f", accel_data[0]));
                ((TextView) AccelerometerActivity.this.findViewById(R.id.accel_y)).setText("Y: " + String.format("%.3f", accel_data[1]));
                ((TextView) AccelerometerActivity.this.findViewById(R.id.accel_z)).setText("Z: " + String.format("%.3f", accel_data[2]));
                ((TextView) AccelerometerActivity.this.findViewById(R.id.pitch)).setText("PITCH: " + String.format("%.1f", pitch));
                ((TextView) AccelerometerActivity.this.findViewById(R.id.roll)).setText("ROLL: " + String.format("%.1f", roll));
                ((TextView) AccelerometerActivity.this.findViewById(R.id.serial_in)).setText("Force/Flex: " + String.format("%.1f", force) + " / " +  String.format("%.1f", flex));
                ((ImageView) AccelerometerActivity.this.findViewById(R.id.microbit)).setRotationX((float) roll);
                ((ImageView) AccelerometerActivity.this.findViewById(R.id.microbit)).setRotationY((float) pitch);
                if (recording) {
                    record_data[next_rec_index][0] = accel_data[0];
                    record_data[next_rec_index][1] = accel_data[1];
                    record_data[next_rec_index][2] = accel_data[2];
                    record_data[next_rec_index][3] = pitch;
                    record_data[next_rec_index][4] = roll;
                    record_data[next_rec_index][5] = force;
                    record_data[next_rec_index][6] = flex;
                    next_rec_index++;
                    if (next_rec_index == SAMPLE_SIZE) {
                        recording = false;
                        if (is_prediction) {
                            ((Button) AccelerometerActivity.this.findViewById(R.id.predict_btn)).setEnabled(true);
                        } else {
                            ((Button) AccelerometerActivity.this.findViewById(R.id.record_btn)).setEnabled(true);
                        }
                        // Log.i(Constants.TAG,"Data: " + gson.toJson(record_data));
                        sendAccelerometerData(gson.toJson(record_data), is_prediction);
                    }
                }
            }
        });
    }

    private void sendAccelerometerData(final String data, final Boolean prediction_mode) {
        Settings settings = Settings.getInstance();
        String server_url = settings.getServer_url();
        if (server_url.length() == 0) {
            showMsg("Invalid Server URL Failed to Send Data....");
            return;
        }
        String url = "http://" + server_url + "/api/upload_train";;
        if (prediction_mode) {
            url = "http://" + server_url + "/api/get_prediction";
        }

        final String g_id = ((EditText) findViewById(R.id.g_id)).getText().toString();
        final TextView textView = (TextView) findViewById(R.id.resp_status);
        textView.setText("Response: Awaiting...");
        // Request a string response from the provided URL.
        StringRequest stringRequest = new StringRequest(Request.Method.POST, url,
                new Response.Listener<String>() {
                    @Override
                    public void onResponse(String response) {
                        // Display the first 500 characters of the response string.
                        try {
                            JSONObject data = new JSONObject(response);
                            String status = data.getString("status");
                            textView.setText("Response: "+ status);
                            Log.i(Constants.TAG,"Response: " + status);
//                            if (is_training) {
////                                String parts[] = status.replaceAll("^\"|\"$", "").split(":");
////                                if (Integer.parseInt(parts[1]) >= 5) {
////                                    ((Button) AccelerometerActivity.this.findViewById(R.id.train_btn)).setEnabled(true);
////                                }
////                            }
                        } catch (JSONException e) {
                            e.printStackTrace();
                        }

                    }
                }, new Response.ErrorListener() {
            @Override
            public void onErrorResponse(VolleyError error) {
                textView.setText("That didn't work!");
            }
        }) {
            @Override
            protected Map<String, String> getParams() throws AuthFailureError {
                Map<String, String> params = new HashMap<String, String>();
                params.put("g_id", g_id);
                params.put("data", data);
                return params;
            }
        };

        // Add the request to the RequestQueue.
                queue.add(stringRequest);
    }


    private void sendTrainRequest() {
        Settings settings = Settings.getInstance();
        String server_url = settings.getServer_url();
        if (server_url.length() == 0) {
            showMsg("Invalid Server URL Failed to Send Data....");
            return;
        }
        String url ="http://" + server_url + "/api/start_train";
        final TextView textView = (TextView) findViewById(R.id.resp_status);
        textView.setText("Response: Awaiting...");
        // Request a string response from the provided URL.
        StringRequest stringRequest = new StringRequest(Request.Method.POST, url,
                new Response.Listener<String>() {
                    @Override
                    public void onResponse(String response) {
                        // Display the first 500 characters of the response string.
                        try {
                            JSONObject data = new JSONObject(response);
                            String status = data.getString("status");
                            textView.setText("Response: "+ status);
                            Log.i(Constants.TAG,"Response: " + status);
                        } catch (JSONException e) {
                            e.printStackTrace();
                        }

                    }
                }, new Response.ErrorListener() {
            @Override
            public void onErrorResponse(VolleyError error) {
                textView.setText("That didn't work!");
            }
        }) {
            @Override
            protected Map<String, String> getParams() throws AuthFailureError {
                Map<String, String> params = new HashMap<String, String>();
                return params;
            }
        };

        // Add the request to the RequestQueue.
        queue.add(stringRequest);
    }

	private void showBenchmark() {
			final int notifications_per_minute = notification_count;
			final int notifications_per_second = notification_count / 60;
			Log.d(Constants.TAG,"Minute: " + Integer.toString(minute_number));
			Log.d(Constants.TAG,"Notification Count: " + Integer.toString(notification_count));
			Log.d(Constants.TAG,"Notifications per Second: " + Integer.toString(notifications_per_second));
	}

    @Override
    public void connectionStatusChanged(boolean connected) {
        if (connected) {
            showMsg(Utility.htmlColorGreen("Connected"));
        } else {
            showMsg(Utility.htmlColorRed("Disconnected"));
            this.toggleButtons(false);
        }
    }

    @Override
    public void serviceDiscoveryStatusChanged(boolean new_state) {

    }
}