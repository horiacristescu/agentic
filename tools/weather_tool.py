from pydantic import BaseModel


class WeatherTool(BaseModel):
    """Get current temperature for a city"""

    city: str

    def execute(self) -> str:
        # Mock weather data
        weather_data = {
            "Romania/Bucharest": "22°C, sunny",
            "USA/New York": "15°C, cloudy",
            "France/Paris": "18°C, rainy",
            "UK/London": "12°C, foggy",
        }

        if self.city not in weather_data:
            raise ValueError(
                f"City '{self.city}' not found in database. Proper format is 'Country/City'."
            )

        return weather_data[self.city]
