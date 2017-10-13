/*
Copyright 2017 The Camlistore Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"net/url"
	"os"
	"path/filepath"

	vision "cloud.google.com/go/vision/apiv1"
	"golang.org/x/net/context"
	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"
	"google.golang.org/api/option"
)

var (
	flagServiceAccount = flag.String("service_account", "", "Path to a service account credentials file")
	flagClientID       = flag.String("client_id", "", "Path to a client ID credentials file")
	flagInput          = flag.String("input", "", "Path to an image with text to be OCRed")
)

// getClient uses a Context and Config to retrieve a Token
// then generate a Client. It returns the generated Client.
func getToken(ctx context.Context, config *oauth2.Config) (*oauth2.Token, error) {
	cacheFile, err := tokenCacheFile()
	if err != nil {
		return nil, fmt.Errorf("unable to get path to cached credential file. %v", err)
	}
	tok, err := tokenFromFile(cacheFile)
	if err != nil {
		tok = getTokenFromWeb(config)
		saveToken(cacheFile, tok)
	}
	return tok, nil
}

// getTokenFromWeb uses Config to request a Token.
// It returns the retrieved Token.
func getTokenFromWeb(config *oauth2.Config) *oauth2.Token {
	authURL := config.AuthCodeURL("state-token", oauth2.AccessTypeOffline)
	fmt.Printf("Go to the following link in your browser then type the "+
		"authorization code: \n%v\n", authURL)

	var code string
	if _, err := fmt.Scan(&code); err != nil {
		log.Fatalf("Unable to read authorization code %v", err)
	}

	tok, err := config.Exchange(oauth2.NoContext, code)
	if err != nil {
		log.Fatalf("Unable to retrieve token from web %v", err)
	}
	return tok
}

// tokenCacheFile generates credential file path/filename.
// It returns the generated credential path/filename.
func tokenCacheFile() (string, error) {
	tokenCacheDir := "./credentials"
	os.MkdirAll(tokenCacheDir, 0700)
	return filepath.Join(tokenCacheDir,
		url.QueryEscape("cache.json")), nil
}

// tokenFromFile retrieves a Token from a given file path.
// It returns the retrieved Token and any read error encountered.
func tokenFromFile(file string) (*oauth2.Token, error) {
	f, err := os.Open(file)
	if err != nil {
		return nil, err
	}
	t := &oauth2.Token{}
	err = json.NewDecoder(f).Decode(t)
	defer f.Close()
	return t, err
}

// saveToken uses a file path to create a file and store the
// token in it.
func saveToken(file string, token *oauth2.Token) {
	fmt.Printf("Saving credential file to: %s\n", file)
	f, err := os.OpenFile(file, os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0600)
	if err != nil {
		log.Fatalf("Unable to cache oauth token: %v", err)
	}
	defer f.Close()
	json.NewEncoder(f).Encode(token)
}

func visionClient(ctx context.Context) (*vision.ImageAnnotatorClient, error) {
	if *flagServiceAccount != "" {
		return vision.NewImageAnnotatorClient(ctx, option.WithCredentialsFile(*flagServiceAccount))
	}

	b, err := ioutil.ReadFile(*flagClientID)
	if err != nil {
		return nil, fmt.Errorf("unable to read client id file: %v", err)
	}
	config, err := google.ConfigFromJSON(b, scopeURLs...)
	if err != nil {
		return nil, fmt.Errorf("unable to parse client id file to config: %v", err)
	}
	tok, err := getToken(ctx, config)
	if err != nil {
		return nil, err
	}
	ts := config.TokenSource(ctx, tok)
	return vision.NewImageAnnotatorClient(ctx, option.WithTokenSource(ts))
}

var scopeURLs = vision.DefaultAuthScopes()

func main() {
	flag.Parse()
	if *flagServiceAccount == "" && *flagClientID == "" {
		log.Fatalf("either -service_account or -client_id must be specified")
	} else if *flagServiceAccount != "" && *flagClientID != "" {
		log.Fatalf("-service_account and -client_id are mutually exclusive")
	}
	if *flagInput == "" {
		log.Fatalf("-input needs to be specified")
	}
	ctx := context.Background()

	cl, err := visionClient(ctx)
	if err != nil {
		log.Fatalf("Failed to create client: %v", err)
	}

	file, err := os.Open(*flagInput)
	if err != nil {
		log.Fatalf("Failed to read file: %v", err)
	}
	defer file.Close()
	image, err := vision.NewImageFromReader(file)
	if err != nil {
		log.Fatalf("Failed to create image entity: %v", err)
	}

	texts, err := cl.DetectTexts(ctx, image, nil, -1)
	if err != nil {
		log.Fatalf("Error detecting text: %v", err)
	}

	fmt.Println("Text:")
	for _, t := range texts {
		fmt.Println(t.Description)
	}
}
