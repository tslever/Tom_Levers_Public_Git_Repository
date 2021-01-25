
// Allows executable to find class MessageHandler.
package com.mycompany.serverutilities.clientcommunicationutilities;

// Imports classes.
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.net.URLDecoder;
import java.util.Arrays;
import java.util.HashMap;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.logging.LogRecord;
import org.json.JSONObject;

/**
 * Defines class MessageHandler whose instances serve as message handlers
 * between message interfaces and controllers.
 * @version 0.0
 * @author Tom Lever
 */
class MessageHandler implements HttpHandler {
    
    private final static Logger logger =
        Logger.getLogger(MessageHandler.class.getName());
    
    private final HashMap<String, Controller> hashMapOfKeysAndControllers;
    
    /**
     * Defines constructor MessageHandler which sets attributes of this message
     * handler with inputs.
     * @param controllerToUse
     */
    public MessageHandler() {
        logger.log(new LogRecord(Level.INFO,
            "MessageHandler constructor: Started."));
        
        this.hashMapOfKeysAndControllers = new HashMap();
    }
    
    public void addController(
        String keyToIdentifyController, Controller controllerToUse) {
        hashMapOfKeysAndControllers.put(
            keyToIdentifyController, controllerToUse);
    }
    
    /**
     * Defines method handle to:
     * 1) Extract the ice cream application message in an HTTP message,
     * 2) Get the body of the ice cream application message,
     * 3) Define an array of strings, each with a key to identify a controller
     *    and values to pass to the controller,
     * 4) Define a hash map of keys and values
     *    (by calling defineHashMapOfKeysAndValues), and
     * 5) Have controllers process values
     *    (by calling haveControllersProcessValues).
     * @param httpExchange
     */
    @Override
    public void handle(HttpExchange httpExchange) {
        logger.log(new LogRecord(Level.INFO,
            "MessageHandler.handle: Started."));
        
        IceCreamApplicationMessage iceCreamApplicationMessage =
            getIceCreamApplicationMessage(httpExchange);
        logger.log(new LogRecord(Level.INFO,
            "MessageHandler.handle: Got ice cream application message from " +
            "HTTP message from client."));
        
        haveControllersProcessValues(
            defineHashMapOfKeysAndValues(
               iceCreamApplicationMessage.getBodyOfClientMessage()),
            this.hashMapOfKeysAndControllers);
        
        // Build a response from haveControllersProcessValues
        // If there was a JSONException, build malformed-JSON response.
        
        // Send the response.
    }
    
    // Create utility to split a string on "&" into an array of any number of
    // strings.
    
    
    /**
     * Defines method defineHashMapOfKeysAndValues to, for each string in
     * keysAndValuesToUse with a key to identify a controller and values to pass
     * to the controller:
     * 1) Convert the string into a two-element array of key and values,
     * 2) Add key and values to a hashMap if the key corresponds to a valid
     *    controller and the key and values haven't already been added to the
     *    hash map, and
     * 3) Adds "invalid-key" and values "{'invalid-key': 'no-valid-keys'}" to
     *    the hash map if no keys and values were added to the hash map in
     *    Step 2.
     * @param keysAndValuesToUse
     * @return
     */
    private HashMap<String, String> defineHashMapOfKeysAndValues(
        String bodyOfClientMessage) {
        
        String[] keysToIdentifyControllersAndValuesToPassToControllers =
            bodyOfClientMessage.split("&");
        logger.log(new LogRecord(Level.INFO,
            "MessageHandler.handle: Split the body of the client message " +
            "based on delimiter '&' into the string array of " +
            "keys+plus+values '" +
            Arrays.toString(
                keysToIdentifyControllersAndValuesToPassToControllers) +
            "'."));
        
        String[] keyToIdentifyControllerAndValuesToPassToControllerAsArray;
        String keyToIdentifyController;
        HashMap<String, String> hashMap = new HashMap();
        String valuesToPassToControllerInURLEncoding;
        
        for (String keyToIdentifyControllerAndValuesToPassToControllerAsString :
            keysToIdentifyControllersAndValuesToPassToControllers) {
            logger.log(new LogRecord(Level.INFO,
                "MessageHandler.handle: Working with the following key to " +
                "identify controller and associated values to pass to " +
                "controller: '" +
                keyToIdentifyControllerAndValuesToPassToControllerAsString +
                "'."));
                           
            keyToIdentifyControllerAndValuesToPassToControllerAsArray =
                keyToIdentifyControllerAndValuesToPassToControllerAsString
                .split("=", 2);
            logger.log(new LogRecord(Level.INFO,
                "MessageHandler.handle: Split the key / values string " +
                "based on delimiter '=' into String array '" +
                Arrays.toString(
                keyToIdentifyControllerAndValuesToPassToControllerAsArray) +
                "'."));

            keyToIdentifyController =
                keyToIdentifyControllerAndValuesToPassToControllerAsArray[0];
            logger.log(new LogRecord(Level.INFO,
                "MessageHandler.handle: Found keyToIdentifyController '" +
                keyToIdentifyController + "'."));
            
            if (!hashMap.containsKey(keyToIdentifyController) &&
                hashMapOfKeysAndControllers.containsKey(
                    keyToIdentifyController) &&
                !keyToIdentifyController.equals("invalid-key")) {
                valuesToPassToControllerInURLEncoding =
                    keyToIdentifyControllerAndValuesToPassToControllerAsArray
                    [1];
                hashMap.put(
                    keyToIdentifyController,
                    valuesToPassToControllerInURLEncoding);
            }
        }
        
        if (hashMap.isEmpty()) {
            hashMap.put("invalid-key", "{'invalid-key': 'no-valid-keys'}");
        }
        
        return hashMap;
    }
    
    /**
     * Defines method haveControllersProcessValues to, for each key in
     * hashMapToUse:
     * 1) Try to decode the values associated with that key from URL encoding
     *    to JSON format,
     * 2) Construct a JSONObject based on the decoded values; and
     * 3) Call the process method of the controller associated with that key,
     *    passing the process method the JSONObject representing values.
     * @param keysAndValuesToUse
     * @return
     */
    private void haveControllersProcessValues(
        HashMap<String, String> hashMapOfKeysAndValues,
        HashMap<String, Controller> hashMapOfKeysAndControllers) {
        
        String valuesToPassToControllerInURLEncoding;
        String valuesToPassToControllerInJSONFormat;
        JSONObject valuesToPassToControllerAsJSONObject;
        
        for (String keyFromHashMapOfKeysAndValues :
             hashMapOfKeysAndValues.keySet()) {
                
            valuesToPassToControllerInURLEncoding =
                hashMapOfKeysAndValues.get(keyFromHashMapOfKeysAndValues);

            try {
                valuesToPassToControllerInJSONFormat = URLDecoder.decode(
                    valuesToPassToControllerInURLEncoding, "UTF-8");
                logger.log(new LogRecord(Level.INFO,
                    "MessageHandler.handle: Decoded values into the " +
                    "String (in JSON format) '" +
                    valuesToPassToControllerInJSONFormat + "'."));

                valuesToPassToControllerAsJSONObject = new JSONObject(
                    valuesToPassToControllerInJSONFormat);
                logger.log(new LogRecord(Level.INFO,
                    "MessageHandler.handle: Converted values to the " +
                    "JSONObject '" +
                    valuesToPassToControllerAsJSONObject.toString() +
                    "'."));

                logger.log(new LogRecord(Level.INFO,
                    "MessageHandler.handle: Calling the process " +
                    "method of the controller associated with key '" +
                    keyFromHashMapOfKeysAndValues + "'. Passing to the " +
                    "process method the JSONObject '" +
                    valuesToPassToControllerAsJSONObject.toString() +
                    "'."));
                hashMapOfKeysAndControllers
                    .get(keyFromHashMapOfKeysAndValues)
                    .process(valuesToPassToControllerAsJSONObject);
            }
            catch (UnsupportedEncodingException e) {
                logger.log(new LogRecord(Level.SEVERE,
                    e.toString() + "\n" +
                    "Caught UnsupportedEncodingException thrown by " +
                    "decoder."));
            }
            
            // TODO: Tell requester that there's an error with their request.
            //catch (JSONException e) {
            //    
            //}
            
            // TODO: Catch NotJSONObjectException.
            catch (Exception e) {
                logger.log(new LogRecord(Level.SEVERE,
                    e.toString() + "\n" +
                    "Caught Exception e thrown by process."));
            }
        } 
    }
    
    /**
     * Defines getIceCreamApplicationMessage, which gets an ice cream
     * application method from an HTTP message associated with an HttpExchange.
     * @param httpExchangeToUse
     * @return 
     */
    private IceCreamApplicationMessage getIceCreamApplicationMessage(
        HttpExchange httpExchangeToUse) {
            logger.log(new LogRecord(Level.INFO,
                "MessageHandler.getIceCreamApplicationMessage: Started."));
        
        String bodyOfClientMessage = "";
        
        // NetBeans wanted me to use a try-with-resources block.
        // Additionally, try block here simplifies method handle.
        try (
            InputStream inputStream = httpExchangeToUse.getRequestBody();
            InputStreamReader inputStreamReader =
                new InputStreamReader(inputStream, "UTF-8");
            BufferedReader bufferedReader =
                new BufferedReader(inputStreamReader);
        ) {
            StringBuilder stringBuilder = new StringBuilder();
            int readByteAsInt;
            while( (readByteAsInt = bufferedReader.read()) != -1 ) {
                stringBuilder.append( (char)readByteAsInt );
            }

           bodyOfClientMessage = stringBuilder.toString();
            logger.log(new LogRecord(Level.INFO,
                "MessageHandler.getIceCreamApplicationMessage: The body of " +
                "the client message is '" + bodyOfClientMessage + "'."));
        }
        catch (IOException e) {
            logger.log(new LogRecord(Level.SEVERE,
                e.toString() + "\n" +
                "Caught IOException thrown by InputStreamReader constructor."));
        }
        
        logger.log(new LogRecord(Level.INFO,
            "MessageHandler.getIceCreamApplicationMessage: Returning new " +
            "ice cream application message based on body of client message."));
        return new IceCreamApplicationMessage(bodyOfClientMessage);
    }
}