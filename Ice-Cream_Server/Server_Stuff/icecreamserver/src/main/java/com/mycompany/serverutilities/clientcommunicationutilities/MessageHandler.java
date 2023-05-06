
// Allows setMessageInterfaces to find class MessageHandler.
package com.mycompany.serverutilities.clientcommunicationutilities;

// Imports classes.
/*import com.
       mycompany.
       serverutilities.
       clientcommunicationutilities.
       IceCreamClientCommunication.
       InputToProcessNotJSONObjectException;*/
import com.mycompany.serverutilities.productutilities.Answer;
import com.mycompany.serverutilities.productutilities.AnswerBuilder;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.IOException;
import java.io.OutputStream;
import java.io.UnsupportedEncodingException;
import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.logging.LogRecord;
import org.json.JSONException;
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
     * Defines constructor MessageHandler, which instantiates this.HashMap.
     */
    public MessageHandler() {
        this.hashMapOfKeysAndControllers = new HashMap();
    }
    
    /**
     * Defines method addController, which adds controllerToUse to a row in
     * this.HashMap with keyToIdentifyController.
     * @param keyToIdentifyController
     * @param controllerToUse 
     */
    public void addController(
        String keyToIdentifyController, Controller controllerToUse) {
        this.hashMapOfKeysAndControllers.put(
            keyToIdentifyController, controllerToUse);
    }
    
    /**
     * Defines method handle to:
     * 1) Extract the ice cream application message in an HTTP message,
     * 2) Log the body of the client message as info,
     * 3) Get the body of the ice cream application message,
     * 4) Define a hash map of keys to identify controllers and values
     *    for controllers to process,
     * 5) Have controllers process values,
     * 6) Build a response from a returned Answer,
     * 7) Log the response, and
     * 8) Send the response to the client.
     * @param httpExchange
     */
    @Override
    public void handle(HttpExchange httpExchange) {
        
        IceCreamApplicationMessage iceCreamApplicationMessage;
        Answer answer;
        try {
            iceCreamApplicationMessage =
                getIceCreamApplicationMessage(httpExchange);
            
            logger.log(new LogRecord(Level.INFO,
                "Body of client message: " +
                iceCreamApplicationMessage.getBodyOfClientMessage()));
            
            answer = haveControllersProcessValues(
                defineHashMapOfKeysAndValues(
                   iceCreamApplicationMessage.getBodyOfClientMessage()),
                this.hashMapOfKeysAndControllers);
        }
        catch (GetIceCreamApplicationMessageRecognizedAHackException e) {
            answer = AnswerBuilder.buildAnswerWithInfo(
                "Zero products available: getIceCreamApplicationMessage " +
                "recognized a hack.");
        }
        catch (UnsupportedEncodingException e) {
            answer = AnswerBuilder.buildAnswerWithInfo(
                "Zero products available: getIceCreamApplicationMessage " +
                "threw an UnsupportedEncodingException.");
        }
        catch (IOException e) {
            answer = AnswerBuilder.buildAnswerWithInfo(
                "Zero products available: getIceCreamApplicationMessage " +
                "threw an IOException.");
        }
        
        byte[] response = buildResponseFrom(answer);

        logger.log(new LogRecord(Level.INFO,
            "Response: " + new String(response, StandardCharsets.UTF_8)));

        send(response, httpExchange);
    }
    
    /**
     * Defines getIceCreamApplicationMessage, which gets an ice cream
     * application message from an HTTP message associated with an HttpExchange.
     * @param httpExchangeToUse
     * @return IceCreamApplicationMessage
     */
    private IceCreamApplicationMessage getIceCreamApplicationMessage(
        HttpExchange httpExchangeToUse)
        throws GetIceCreamApplicationMessageRecognizedAHackException,
               UnsupportedEncodingException,
               IOException {
        
        boolean getIceCreamApplicationMessageRecognizedHack = false;
        if (getIceCreamApplicationMessageRecognizedHack) {
            throw new GetIceCreamApplicationMessageRecognizedAHackException(
                "getIceCreamApplicationMessage recognized hack.");
        }
        
        String bodyOfClientMessage;
        
        InputStream inputStream = httpExchangeToUse.getRequestBody();

        // NetBeans wants to split declaration and assignment.
        InputStreamReader inputStreamReader;
        inputStreamReader = new InputStreamReader(inputStream, "UTF-8");
        
        BufferedReader bufferedReader =
            new BufferedReader(inputStreamReader);

        StringBuilder stringBuilder = new StringBuilder();
        int readByteAsInt;
        while( (readByteAsInt = bufferedReader.read()) != -1 ) {
            stringBuilder.append( (char)readByteAsInt );
        }

        bodyOfClientMessage = stringBuilder.toString();
        
        return new IceCreamApplicationMessage(bodyOfClientMessage);
    }
    
    /**
     * Defines method defineHashMapOfKeysAndValues to:
     * 1) Split bodyOfClientMessage into
     * keysToIdentifyControllersAndValuesToPassToControllers based on "&",
     * 2) For each keyToIdentifyControllerAndValuesToPAssToControllerAsString,
     * split the string into
     * keyToIdentifyControllerAndValuesToPassToControllerAsArray, and
     * 3) If the keyToIdentifyController is not already in the hashMap being
     * defined and this.hashMapOfKeysAndControllers contains
     * keyToIdentifyController, then put in the hashMap being defined at
     * the keyToIdentifyController the values to pass to the controller.
     * @param bodyOfClientMessage
     * @return hashMap
     */
    private HashMap<String, String> defineHashMapOfKeysAndValues(
        String bodyOfClientMessage) {
        
        String[] keysToIdentifyControllersAndValuesToPassToControllers =
            bodyOfClientMessage.split("&");
        
        String[] keyToIdentifyControllerAndValuesToPassToControllerAsArray;
        String keyToIdentifyController;
        HashMap<String, String> hashMap = new HashMap();
        String valuesToPassToController;
        
        for (String keyToIdentifyControllerAndValuesToPassToControllerAsString :
            keysToIdentifyControllersAndValuesToPassToControllers) {
                           
            keyToIdentifyControllerAndValuesToPassToControllerAsArray =
                keyToIdentifyControllerAndValuesToPassToControllerAsString
                .split("=", 2);

            keyToIdentifyController =
                keyToIdentifyControllerAndValuesToPassToControllerAsArray[0];
            
            if (!hashMap.containsKey(keyToIdentifyController) &&
                this.hashMapOfKeysAndControllers.containsKey(
                    keyToIdentifyController)) {
                valuesToPassToController =
                    keyToIdentifyControllerAndValuesToPassToControllerAsArray
                    [1];
                hashMap.put(
                    keyToIdentifyController,
                    valuesToPassToController);
            }
        }
        
        return hashMap;
    }
    
    /**
     * Defines method haveControllersProcessValues to, if 'search-parameters' is
     * in hashMapOfKeysAndValues:
     * 1) Decode the values corresponding to key 'search-parameters' if the
     * values are in format "x-www-form-urlencoded",
     * 2) Construct a JSONObject based on the decoded values, and
     * 3) Have the controller associated with the key 'search-parameters'
     * process the valuesToPassToControllerAsJSONObject.
     * @param hashMapOfKeysAndValues, hashMapOfKeysAndControllers
     * @return Answer
     */
    private Answer haveControllersProcessValues(
        HashMap<String, String> hashMapOfKeysAndValues,
        HashMap<String, Controller> hashMapOfKeysAndControllers) {
        
        String searchParameters = "search-parameters";
        
        if (!hashMapOfKeysAndValues.keySet().contains(searchParameters)) {
            return AnswerBuilder.buildAnswerWithInfo(
                "Zero products available: Body of client message does not " +
                "contain key 'search-parameters'.");
        }
        
        String valuesToPassToController =
            hashMapOfKeysAndValues.get(searchParameters);

        try {
            String valuesToPassToControllerInJSONFormat = URLDecoder.decode(
                valuesToPassToController, "UTF-8");
            
            JSONObject valuesToPassToControllerAsJSONObject = new JSONObject(
                valuesToPassToControllerInJSONFormat);
            
            return hashMapOfKeysAndControllers
                .get(searchParameters)
                .process(valuesToPassToControllerAsJSONObject);
        }
        catch (UnsupportedEncodingException e) {
            return AnswerBuilder.buildAnswerWithInfo(
                "Zero products available: Unable to decode " +
                "valuesToPassToController.");
        }
        catch (JSONException e) {
            return AnswerBuilder.buildAnswerWithInfo(
                "Zero products available: JSON associated with " +
                "key 'search-parameters' is invalid.");
        }
        catch (ProcessException e) {
            return AnswerBuilder.buildAnswerWithInfo(
                "Zero products available: The controller associated with " +
                "key 'search-parameters' was passed an Object that was not " +
                "a JSONObject.");
        }
    }
    
    /**
     * Defines method buildResponseFrom to build and return a byte[] response
     * based on an Answer.
     * @param answerToUse
     * @return response.getBytes()
     */
    private byte[] buildResponseFrom(Answer answerToUse) {
        
        String response = "{";
        
        HashMap<String, String> hashMap = answerToUse.getHashMap();
        
        int numberOfKeys = hashMap.keySet().size();
        String[] keys = new String[numberOfKeys];
        int present_index = 0;
        for (String key : hashMap.keySet()) {
            keys[present_index] = key;
            present_index++;
        }
        for (int i = 0; i < numberOfKeys-1; i++) {
            response +=
                "\"" + keys[i] + "\": " + hashMap.get(keys[i]) + ", ";            
        }
        response +=
            "\"" + keys[numberOfKeys-1] + "\": " +
            hashMap.get(keys[numberOfKeys-1]);
        response += "}";
        
        return response.getBytes();
    }
    
    /**
     * Defines method send to send a response back to the ice cream client.
     * @param response
     * @param httpExchange
     */
    private void send(byte[] response, HttpExchange httpExchange) {
    
        try (OutputStream outputStream = httpExchange.getResponseBody();) {
            // Both getResponseBody and sendResponseHeaders throw IOExceptions.
            //OutputStream outputStream = httpExchange.getResponseBody();
            httpExchange.sendResponseHeaders(200, response.length);
            outputStream.write(response);
            //outputStream.close();
        }
        catch (IOException e) {
            try {
                // Send response headers for Internal Server Error. Send no
                // response body.
                httpExchange.sendResponseHeaders(500, -1);
            }
            catch (IOException f) {
                
            }
        }
    }
}