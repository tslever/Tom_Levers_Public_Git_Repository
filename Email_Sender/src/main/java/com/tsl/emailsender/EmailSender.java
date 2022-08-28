package com.tsl.emailsender;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.GeneralSecurityException;
import java.util.Collections;
import java.util.List;
import java.util.Properties;

import javax.activation.DataHandler;
import javax.activation.FileDataSource;
import javax.mail.MessagingException;
import javax.mail.Multipart;
import javax.mail.Session;
import javax.mail.internet.InternetAddress;
import javax.mail.internet.MimeBodyPart;
import javax.mail.internet.MimeMessage;
import javax.mail.internet.MimeMultipart;

import org.apache.commons.codec.binary.Base64;

import com.google.api.client.auth.oauth2.Credential;
import com.google.api.client.extensions.java6.auth.oauth2.AuthorizationCodeInstalledApp;
import com.google.api.client.extensions.jetty.auth.oauth2.LocalServerReceiver;
import com.google.api.client.googleapis.auth.oauth2.GoogleAuthorizationCodeFlow;
import com.google.api.client.googleapis.auth.oauth2.GoogleClientSecrets;
import com.google.api.client.googleapis.javanet.GoogleNetHttpTransport;
import com.google.api.client.googleapis.json.GoogleJsonError;
import com.google.api.client.googleapis.json.GoogleJsonResponseException;
import com.google.api.client.http.javanet.NetHttpTransport;
import com.google.api.client.json.JsonFactory;
import com.google.api.client.json.gson.GsonFactory;
import com.google.api.client.util.store.FileDataStoreFactory;
import com.google.api.services.gmail.Gmail;
import com.google.api.services.gmail.GmailScopes;
import com.google.api.services.gmail.model.Message;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

public class EmailSender {

    Configurations configurations;
    
    public EmailSender(Configurations configurationsToUse) {
        configurations = configurationsToUse;
    }
    
    public void send() throws GeneralSecurityException, IOException, MessagingException {
    
        // See https://developers.google.com/gmail/api/guides/sending
        // Create email content.
        Path path = Path.of(configurations.emailsDirectory + "/" + configurations.email + "/header.json");
        String headerAsString = Files.readString(path, StandardCharsets.UTF_8);
        JsonObject header = JsonParser.parseString(headerAsString).getAsJsonObject();
        path = Path.of(configurations.emailsDirectory + "/" + configurations.email + "/body.txt");
        String body = Files.readString(path, StandardCharsets.UTF_8);
        
        // Encode as MIME message
        Properties properties = new Properties();
        Session session = Session.getDefaultInstance(properties, null);
        MimeMessage mimeMessage = new MimeMessage(session);
        String subject = header.get("subject").getAsString();
        mimeMessage.setSubject(subject);
        String fromAddress = header.get("from").getAsString();
        mimeMessage.setFrom(fromAddress);
        JsonArray jsonArrayOfToAddresses = header.get("to").getAsJsonArray();
        for (int i = 0; i < jsonArrayOfToAddresses.size(); i++) {
            String toAddress = jsonArrayOfToAddresses.get(i).getAsString();
            InternetAddress toInternetAddress = new InternetAddress(toAddress);
            mimeMessage.addRecipient(javax.mail.Message.RecipientType.TO, toInternetAddress);
        }
        
        MimeBodyPart mimeBodyPart = new MimeBodyPart();
        mimeBodyPart.setContent(body, "text/plain");
        Multipart multipart = new MimeMultipart();
        multipart.addBodyPart(mimeBodyPart);
        File directoryForAttachments = new File(configurations.emailsDirectory + "/" + configurations.email + "/attachments");
        for (File attachment : directoryForAttachments.listFiles()) {
            if (!attachment.isDirectory()) {
                mimeBodyPart = new MimeBodyPart();
                FileDataSource fileDataSource = new FileDataSource(attachment);
                mimeBodyPart.setDataHandler(new DataHandler(fileDataSource));
                mimeBodyPart.setFileName(attachment.getName());
                multipart.addBodyPart(mimeBodyPart);
            }
        }
        mimeMessage.setContent(multipart);
        
        // Encode and wrap the MIME message into a Gmail message
        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
        mimeMessage.writeTo(byteArrayOutputStream);
        byte[] arrayOfBytes = byteArrayOutputStream.toByteArray();
        String encodedEmail = Base64.encodeBase64URLSafeString(arrayOfBytes);
        Message message = new Message();
        message.setRaw(encodedEmail);
        
        // See https://developers.google.com/gmail/api/quickstart/java
        // Build a new authorized API client service.
        final NetHttpTransport networkHttpTransport = GoogleNetHttpTransport.newTrustedTransport();
        JsonFactory jsonFactory = GsonFactory.getDefaultInstance();
        Credential credential = getCredentials(fromAddress, networkHttpTransport, jsonFactory, configurations.credentialsDirectory);
        Gmail.Builder gmailBuilder = new Gmail.Builder(networkHttpTransport, jsonFactory, credential);
        gmailBuilder.setApplicationName("email-sender");
        Gmail gmail = gmailBuilder.build();
        
        // See https://developers.google.com/gmail/api/guides/sending
        try {
            // Send message
            message = gmail.users().messages().send(fromAddress, message).execute();
        } catch (GoogleJsonResponseException e) {
            // TODO(developer) - handle error appropriately
            GoogleJsonError error = e.getDetails();
            if (error.getCode() == 403) {
                System.err.println("Unable to send message: " + e.getDetails());
            } else {
                throw e;
            }
        }
    }
    
    /**
     * Creates an authorized Credential object.
     * 
     * @param httpTransport, a network HTTP Transport.
     * @return an authorized Credential object.
     * @throws IOException if the oauth2ClientCredentials.json file cannot be found.
     */
    private static Credential getCredentials(String fromAddress, final NetHttpTransport httpTransport, JsonFactory jsonFactory, String pathToDirectoryContainingCredentials) throws IOException {
        
        // Download oauth2ClientCredentials.json from https://console.cloud.google.com/apis/credentials?project=<name-of-google-cloud-project>.
        FileReader fileReader = new FileReader(pathToDirectoryContainingCredentials + "/" + fromAddress + "/oauth2ClientCredentials.json");
        GoogleClientSecrets googleClientSecrets = GoogleClientSecrets.load(jsonFactory, fileReader);
        String scopeForSendingEmail = GmailScopes.GMAIL_SEND;
        List<String> listOfScopes = Collections.singletonList(scopeForSendingEmail);
        GoogleAuthorizationCodeFlow.Builder googleAuthorizationCodeFlowBuilder = new GoogleAuthorizationCodeFlow.Builder(httpTransport, jsonFactory, googleClientSecrets, listOfScopes);
        File directoryForAccessToken = new File(pathToDirectoryContainingCredentials + "/" + fromAddress);
        FileDataStoreFactory fileDataStoreFactory = new FileDataStoreFactory(directoryForAccessToken);
        googleAuthorizationCodeFlowBuilder.setDataStoreFactory(fileDataStoreFactory);
        googleAuthorizationCodeFlowBuilder.setAccessType("offline");
        GoogleAuthorizationCodeFlow googleAuthorizationCodeFlow = googleAuthorizationCodeFlowBuilder.build();
        LocalServerReceiver.Builder localServerReceiverBuilder = new LocalServerReceiver.Builder();
        localServerReceiverBuilder.setPort(8888);
        LocalServerReceiver localServerReceiver = localServerReceiverBuilder.build();
        AuthorizationCodeInstalledApp authorizationCodeInstalledApp = new AuthorizationCodeInstalledApp(googleAuthorizationCodeFlow, localServerReceiver);
        Credential credential = authorizationCodeInstalledApp.authorize("user");
        return credential;
    }
}