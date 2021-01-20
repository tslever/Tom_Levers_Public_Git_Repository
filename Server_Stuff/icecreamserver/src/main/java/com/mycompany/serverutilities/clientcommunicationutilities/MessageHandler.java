
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
import java.util.ArrayList;
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
     * Defines method handle to pick up an HTTP message from a message
     * interface, extract the ice cream application message in that HTTP
     * message, and hand sets of parameters in that message to different
     * controllers.
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
            "HTTP message from client. The body of the client message is '" +
            iceCreamApplicationMessage.getBodyOfClientMessage() + "'."));
        
        String[] keysToIdentifyControllersAndValuesToPassToControllers =
            iceCreamApplicationMessage.getBodyOfClientMessage().split("&");
        logger.log(new LogRecord(Level.INFO,
            "MessageHandler.handle: Split the body of the client message " +
            "based on delimiter '&' into the string array of " +
            "keys+plus+values '" +
            Arrays.toString(
                keysToIdentifyControllersAndValuesToPassToControllers) +
            "'."));       
        
        String[] keyToIdentifyControllerAndValuesToPassToControllerAsArray;
        String keyToIdentifyController;
        HashMap<String, String> hashMapOfKeysAndValues = new HashMap();
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
            
            if (!hashMapOfKeysAndValues.containsKey(keyToIdentifyController) &&
                hashMapOfKeysAndControllers.containsKey(
                    keyToIdentifyController) &&
                !keyToIdentifyController.equals("invalid-key")) {
                valuesToPassToControllerInURLEncoding =
                    keyToIdentifyControllerAndValuesToPassToControllerAsArray
                    [1];
                hashMapOfKeysAndValues.put(
                    keyToIdentifyController,
                    valuesToPassToControllerInURLEncoding);
            }
        }
        
        if (hashMapOfKeysAndValues.size() == 0) {
            hashMapOfKeysAndValues.put(
                "invalid-key", "{'invalid-key': 'no-valid-keys'}");
        }
            
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
                this.hashMapOfKeysAndControllers
                    .get(keyFromHashMapOfKeysAndValues)
                    .process(valuesToPassToControllerAsJSONObject);
            }
            catch (UnsupportedEncodingException e) {
                logger.log(new LogRecord(Level.SEVERE,
                    e.toString() + "\n" +
                    "Caught UnsupportedEncodingException thrown by " +
                    "decoder."));
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