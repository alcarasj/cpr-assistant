/* jshint esversion: 6 */

import React from "react";
import { 
  StyleSheet,
  Text,
  View,
  Button
} from "react-native";

export default class CameraScreen extends React.Component {

	static navigationOptions = {
		drawerLabel: "Camera"
	};

	render = () => {
		return (
			<View style={styles.container}>
				<Text>Welcome to the camera screen.</Text>
				<Button
				title="Go to home screen"
				onPress={ () => this.props.navigation.navigate("Home") }
				/>
				<Button
				title="Toggle drawer"
				onPress={ () => this.props.navigation.toggleDrawer() }
				/>
			</View>
		);
	};
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#fff",
    alignItems: "center",
    justifyContent: "center",
  }
});