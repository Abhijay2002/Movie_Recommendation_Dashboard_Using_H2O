from h2o_wave import main, app, Q, ui
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
import random

# Reduced dummy data with only 10 movies for faster processing
dummy_movies = [
    {'id': 1, 'title': 'Action Hero', 'genre': 'Action', 'rating': 4.5, 'year': 2021, 'director': 'James Cameron', 'popularity': 85},
    {'id': 2, 'title': 'Drama Queen', 'genre': 'Drama', 'rating': 4.2, 'year': 2020, 'director': 'Steven Spielberg', 'popularity': 75},
    {'id': 3, 'title': 'Laugh Out Loud', 'genre': 'Comedy', 'rating': 3.8, 'year': 2022, 'director': 'Judd Apatow', 'popularity': 68},
    {'id': 4, 'title': 'Action Reloaded', 'genre': 'Action', 'rating': 4.7, 'year': 2019, 'director': 'Christopher Nolan', 'popularity': 92},
    {'id': 5, 'title': 'Romance in Paris', 'genre': 'Romance', 'rating': 4.1, 'year': 2023, 'director': 'Nancy Meyers', 'popularity': 79},
    {'id': 6, 'title': 'Space Adventures', 'genre': 'Sci-Fi', 'rating': 4.6, 'year': 2022, 'director': 'Denis Villeneuve', 'popularity': 88},
    {'id': 7, 'title': 'Haunted Night', 'genre': 'Horror', 'rating': 3.5, 'year': 2021, 'director': 'Jordan Peele', 'popularity': 72},
    {'id': 8, 'title': 'Mystery Manor', 'genre': 'Mystery', 'rating': 4.3, 'year': 2020, 'director': 'David Fincher', 'popularity': 81},
    {'id': 9, 'title': 'Galactic Wars', 'genre': 'Sci-Fi', 'rating': 4.8, 'year': 2022, 'director': 'Ridley Scott', 'popularity': 94},
    {'id': 10, 'title': 'Heartbreak Boulevard', 'genre': 'Drama', 'rating': 4.0, 'year': 2021, 'director': 'Greta Gerwig', 'popularity': 76}
]

# Generate user viewing history data - simplified for quicker processing
def generate_user_data(num_users=20, movies=dummy_movies):
    user_data = []
    for user_id in range(1, num_users + 1):
        # Simulate user preferences (biased towards specific genres)
        preferred_genres = random.sample(
            ['Action', 'Drama', 'Comedy', 'Sci-Fi', 'Horror', 'Romance', 'Mystery'], 
            random.randint(1, 2)
        )
        # View between 3 and 5 movies
        num_views = random.randint(3, min(5, len(movies)))
        viewed_movies = random.sample(movies, num_views)
        # Bias the ratings based on genre preference
        for movie in viewed_movies:
            rating_bias = 1.0 if movie['genre'] in preferred_genres else 0
            user_rating = min(5.0, max(1.0, movie['rating'] + rating_bias - random.uniform(0, 1.5)))
            user_data.append({
                'user_id': user_id,
                'movie_id': movie['id'],
                'rating': round(user_rating, 1),
                'timestamp': f"2023-{random.randint(1, 6)}-{random.randint(1, 28)}"
            })
    return user_data

# Create and train the SVD recommendation model
def create_recommendation_model(movies, user_data):
    movies_df = pd.DataFrame(movies)
    ratings_df = pd.DataFrame(user_data)
    
    # Create a user-movie (utility) matrix
    user_movie_matrix = ratings_df.pivot_table(
        index='user_id', 
        columns='movie_id', 
        values='rating',
        fill_value=0
    ).values
    
    users = sorted(ratings_df['user_id'].unique())
    movie_ids = sorted(ratings_df['movie_id'].unique())
    idx_to_movie_id = {i: movie_id for i, movie_id in enumerate(movie_ids)}
    movie_id_to_idx = {movie_id: i for i, movie_id in enumerate(movie_ids)}
    
    # Check if the matrix is large enough for SVD
    if min(user_movie_matrix.shape) <= 1:
        print("Matrix too small for SVD, using simple popularity-based recommendations")
        svd = None
        latent_matrix = None
        reconstructed_matrix = None
        components = None
    else:
        n_components = min(2, min(user_movie_matrix.shape) - 1)
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        try:
            latent_matrix = svd.fit_transform(user_movie_matrix)
            reconstructed_matrix = np.dot(latent_matrix, svd.components_)
            components = svd.components_
            print(f"SVD explained variance: {svd.explained_variance_ratio_.sum():.2f}")
        except Exception as e:
            print(f"SVD failed: {e}")
            svd = None
            latent_matrix = None
            reconstructed_matrix = None
            components = None
    
    return {
        'model': svd,
        'movies_df': movies_df,
        'user_movie_matrix': user_movie_matrix,
        'reconstructed_matrix': reconstructed_matrix,
        'latent_matrix': latent_matrix,
        'components': components,
        'users': users,
        'movie_ids': movie_ids,
        'idx_to_movie_id': idx_to_movie_id,
        'movie_id_to_idx': movie_id_to_idx
    }

# Function to get recommendations based on genre and/or ML-based inference
def get_recommendations(genre, model_data, num_recommendations=5):
    movies_df = model_data['movies_df']
    
    # If a genre is specified, filter movies by genre and sort by a computed score
    if genre:
        genre_mask = movies_df['genre'].str.lower() == genre.lower()
        if genre_mask.any():
            filtered_movies = movies_df[genre_mask].copy()
            filtered_movies['score'] = filtered_movies['rating'] * 0.7 + (filtered_movies['popularity'] / 100) * 0.3
            return filtered_movies.sort_values('score', ascending=False).head(num_recommendations).to_dict('records')
    
    # If the SVD model is missing or its reconstruction is unavailable, fallback to top ratings
    if model_data.get('model') is None or model_data.get('reconstructed_matrix') is None:
        return movies_df.sort_values('rating', ascending=False).head(num_recommendations).to_dict('records')
    
    users = model_data.get('users', [])
    if not users:
        return movies_df.sort_values('popularity', ascending=False).head(num_recommendations).to_dict('records')
    
    try:
        # Choose a random user from the training data
        random_user_idx = random.randint(0, len(users) - 1)
        user_id = users[random_user_idx]
        user_idx = users.index(user_id)
        user_ratings = model_data['reconstructed_matrix'][user_idx]
        
        movie_scores = []
        for i, score in enumerate(user_ratings):
            movie_id = model_data['idx_to_movie_id'].get(i)
            if movie_id:
                movie_scores.append((movie_id, score))
        movie_scores.sort(key=lambda x: x[1], reverse=True)
        top_movie_ids = [movie_id for movie_id, _ in movie_scores[:num_recommendations]]
        recommendations = movies_df[movies_df['id'].isin(top_movie_ids)].to_dict('records')
    except Exception as e:
        print(f"Error getting SVD recommendations: {e}")
        recommendations = movies_df.sort_values('rating', ascending=False).head(num_recommendations).to_dict('records')
    
    return recommendations

# Generate user data and create recommendation model
user_data = generate_user_data()
recommendation_model = create_recommendation_model(dummy_movies, user_data)

@app('/movie-dashboard')
async def serve(q: Q):
    """
    Advanced H2O Wave app with ML recommendations demonstrating:
      - Responsive layouts with multiple zones
      - Interactive filtering and sorting
      - ML-based movie recommendations
      - Data visualization
    """
    # Initialize app state and UI layout
    if not q.client.initialized:
        q.page['meta'] = ui.meta_card(
            box='',
            title='Movie Streaming Dashboard',
            theme='h2o-dark',
            icon='VideoLibrary',
            layouts=[
                ui.layout(
                    breakpoint='xs',
                    zones=[
                        ui.zone('header', size='80px'),
                        ui.zone('subheader', size='60px'),
                        ui.zone('content', direction=ui.ZoneDirection.ROW, zones=[
                            ui.zone('sidebar', size='300px'),
                            ui.zone('main', zones=[
                                ui.zone('top_row', direction=ui.ZoneDirection.ROW, zones=[
                                    ui.zone('stats1', size='33%'),
                                    ui.zone('stats2', size='33%'),
                                    ui.zone('stats3', size='33%'),
                                ]),
                                ui.zone('recommendations'),
                                ui.zone('explore'),
                            ]),
                        ]),
                        ui.zone('footer', size='50px'),
                    ]
                )
            ]
        )

        # Header card
        q.page['header'] = ui.header_card(
            box='header',
            title='Movie Streaming Dashboard',
            subtitle='Real-Time ML Recommendations and Trends',
            icon='Movie',
            icon_color='#F39C12',
        )

        # Navigation tab bar
        q.page['nav'] = ui.tab_card(
            box='subheader',
            items=[
                ui.tab(name='#recommendations', label='Recommendations', icon='Star'),
                ui.tab(name='#trending', label='Trending', icon='Trending'),
                ui.tab(name='#genres', label='Genres', icon='Filter'),
                ui.tab(name='#about', label='About', icon='Info'),
            ],
            value='#recommendations'
        )

        # Sidebar with filter options
        q.page['sidebar'] = ui.form_card(
            box='sidebar',
            title='Find Movies',
            items=[
                ui.dropdown(
                    name='genre',
                    label='Genre',
                    placeholder='Select a genre',
                    choices=[
                        ui.choice(name=g, label=g) for g in sorted({m['genre'] for m in dummy_movies})
                    ]
                ),
                ui.slider(name='min_rating', label='Minimum Rating', min=1.0, max=5.0, value=3.0, step=0.1),
                ui.slider(name='min_year', label='Minimum Release Year', min=2019, max=2023, value=2019, step=1),
                ui.slider(name='max_year', label='Maximum Release Year', min=2019, max=2023, value=2023, step=1),
                ui.separator(),
                ui.button(name='submit', label='Get Recommendations', primary=True, icon='WavingHand'),
                ui.button(name='reset', label='Reset Filters', icon='Delete'),
                ui.separator(),
                ui.text('SVD-Powered Recommendations'),
                ui.toggle(name='use_ml', label='Use SVD Model', value=True),
                ui.text_m("Our SVD model analyzes the user-movie rating matrix to predict movies you might enjoy."),
            ]
        )

        # Stats cards simulated with markdown_card
        q.page['stats1'] = ui.markdown_card(
            box='stats1',
            title='Total Movies',
            content=f"**{len(dummy_movies)}** in database"
        )
        avg_rating = sum(m['rating'] for m in dummy_movies) / len(dummy_movies)
        q.page['stats2'] = ui.markdown_card(
            box='stats2',
            title='Average Rating',
            content=f"**{avg_rating:.1f}** across all movies"
        )
        new_releases = sum(1 for m in dummy_movies if m['year'] >= 2022)
        q.page['stats3'] = ui.markdown_card(
            box='stats3',
            title='New Releases',
            content=f"**{new_releases}** from 2022-2023"
        )

        # Initial recommendations display
        q.page['recommendations'] = ui.markdown_card(
            box='recommendations',
            title='🎬 Welcome to Movie Streaming!',
            content='''
            ### Get personalized movie recommendations
            
            Use the filters on the left to discover movies tailored to your preferences.
            Our machine learning algorithm can suggest titles based on what similar users enjoyed.
            
            #### Featured genres:
            - Action
            - Drama
            - Sci-Fi
            - Horror
            '''
        )

        # Data exploration card
        q.page['explore'] = ui.form_card(
            box='explore',
            title='Top Rated Movies',
            items=[
                ui.text_xl('Highest Rated Films in Our Collection'),
                ui.table(
                    name='top_movies',
                    columns=[
                        ui.table_column(name='title', label='Title'),
                        ui.table_column(name='genre', label='Genre'),
                        ui.table_column(name='rating', label='Rating'),
                        ui.table_column(name='year', label='Year'),
                    ],
                    rows=[
                        ui.table_row(
                            name=f'movie_{m["id"]}',
                            cells=[m['title'], m['genre'], str(m['rating']), str(m['year'])]
                        )
                        for m in sorted(dummy_movies, key=lambda x: x['rating'], reverse=True)[:5]
                    ],
                )
            ]
        )

        # Footer card
        q.page['footer'] = ui.footer_card(
            box='footer',
            caption='© 2025 Movie Streaming Inc. | Enhanced with H2O Wave and Machine Learning'
        )

        q.client.initialized = True
        await q.page.save()
        return

    # Reset filters if requested
    if q.args.reset:
        for comp in q.page['sidebar'].items:
            if hasattr(comp, 'name'):
                if comp.name == 'genre':
                    comp.value = None
                elif comp.name == 'min_rating':
                    comp.value = 3.0
                elif comp.name == 'min_year':
                    comp.value = 2019
                elif comp.name == 'max_year':
                    comp.value = 2023
        await q.page.save()
        return

    # Handle recommendations on submit
    if q.args.submit:
        genre = q.args.genre
        min_rating = float(q.args.min_rating) if q.args.min_rating is not None else 3.0
        min_year = int(q.args.min_year) if q.args.min_year is not None else 2019
        max_year = int(q.args.max_year) if q.args.max_year is not None else 2023
        use_ml = q.args.use_ml

        if use_ml and not genre:
            recommendations = get_recommendations(None, recommendation_model)
            title = "🤖 AI-Powered Recommendations"
        else:
            filtered = [
                m for m in dummy_movies 
                if (not genre or m['genre'].lower() == genre.lower()) and
                   m['rating'] >= min_rating and
                   min_year <= m['year'] <= max_year
            ]
            recommendations = sorted(filtered, key=lambda x: x['rating'], reverse=True)
            title = f"🎭 Top {genre} Movies" if genre else "🔍 Movies Matching Your Criteria"

        if recommendations:
            items = [ui.text_xl(title)]
            for i, movie in enumerate(recommendations[:5]):
                items.append(ui.separator())
                items.append(ui.inline([
                    ui.text_l(f"{i+1}. {movie['title']} ({movie['year']})"),
                    ui.text_xl(f"{'★' * int(round(movie['rating']))}{' ☆' * (5 - int(round(movie['rating'])))}")
                ]))
                items.append(ui.text(f"Genre: {movie['genre']} | Director: {movie['director']} | Popularity: {movie['popularity']}%"))
                items.append(ui.progress(
                    label='User Rating',
                    caption=f"{movie['rating']}/5.0",
                    value=movie['rating'] / 5.0,
                ))
            items.append(ui.separator())
            if use_ml and not genre:
                items.append(ui.text_s("These recommendations are powered by our SVD model that identifies latent factors in user ratings."))
            else:
                items.append(ui.text_s("These recommendations are based on your filter criteria."))
            items.append(ui.inline([
                ui.button(name='watch_later', label='My List', icon='Add'),
                ui.button(name='show_similar', label='Similar Movies', icon='BulletedTreeList'),
                ui.button(name='surprise_me', label='Surprise Me', icon='Sparkle', primary=True),
            ]))
            q.page['recommendations'] = ui.form_card(
                box='recommendations',
                title='Personalized Recommendations',
                items=items
            )
        else:
            q.page['recommendations'] = ui.markdown_card(
                box='recommendations',
                title='No Results Found',
                content=f'''
                ## No movies match your filters
                
                Adjust your criteria:
                - Genre: {genre or "None selected"}
                - Minimum Rating: {min_rating}
                - Year Range: {min_year} - {max_year}
                
                Or try our AI recommendations by toggling "Use SVD Model" in the sidebar.
                '''
            )

        # Update the exploration table with filtered movies
        q.page['explore'] = ui.form_card(
            box='explore',
            title='Explore Movies',
            items=[
                ui.text_xl(f'{"All" if not genre else genre} Movies ({min_year}-{max_year})'),
                ui.table(
                    name='all_movies',
                    columns=[
                        ui.table_column(name='title', label='Title'),
                        ui.table_column(name='genre', label='Genre'),
                        ui.table_column(name='year', label='Year'),
                        ui.table_column(name='rating', label='Rating'),
                        ui.table_column(name='director', label='Director'),
                    ],
                    rows=[
                        ui.table_row(
                            name=f'movie_{m["id"]}',
                            cells=[m['title'], m['genre'], str(m['year']), str(m['rating']), m['director']]
                        )
                        for m in dummy_movies if (
                            (not genre or m['genre'].lower() == genre.lower()) and
                            m['rating'] >= min_rating and
                            min_year <= m['year'] <= max_year
                        )
                    ],
                )
            ]
        )
    await q.page.save()
