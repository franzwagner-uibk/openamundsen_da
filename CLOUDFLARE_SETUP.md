# Cloudflare Pages Deployment Setup

This guide will help you set up automatic deployment of your documentation to Cloudflare Pages using GitHub Actions.

## Prerequisites

- A Cloudflare account
- Access to your GitHub repository settings
- Your repository must be private (already the case)

## Step 1: Get Your Cloudflare Account ID

1. Log in to your [Cloudflare Dashboard](https://dash.cloudflare.com/)
2. Click on "Workers & Pages" in the left sidebar
3. Your **Account ID** is displayed in the right sidebar under "Account details"
   - It looks like: `39ec0bf0b6512b47bfbc3d71225c626d` (you already have this!)

## Step 2: Create a Cloudflare API Token

1. Go to [Cloudflare API Tokens](https://dash.cloudflare.com/profile/api-tokens)
2. Click **"Create Token"**
3. Click **"Use template"** next to "Edit Cloudflare Workers"
4. Or create a custom token with these permissions:
   - **Account** → **Cloudflare Pages** → **Edit**
5. Under "Account Resources":
   - Include → Your account
6. Click **"Continue to summary"**
7. Click **"Create Token"**
8. **IMPORTANT**: Copy the token immediately - you won't see it again!

## Step 3: Add Secrets to GitHub

1. Go to your GitHub repository: `https://github.com/franzwagner-uibk/openamundsen_da`
2. Click **Settings** → **Secrets and variables** → **Actions**
3. Click **"New repository secret"**
4. Add the following two secrets:

### Secret 1: CLOUDFLARE_ACCOUNT_ID
   - Name: `CLOUDFLARE_ACCOUNT_ID`
   - Value: `39ec0bf0b6512b47bfbc3d71225c626d` (your Account ID)

### Secret 2: CLOUDFLARE_API_TOKEN
   - Name: `CLOUDFLARE_API_TOKEN`
   - Value: The API token you created in Step 2

## Step 4: Create Cloudflare Pages Project (First Time Only)

The first time you deploy, you need to create the Pages project:

1. Go to [Cloudflare Dashboard](https://dash.cloudflare.com/)
2. Click **Workers & Pages** → **Create application** → **Pages**
3. Click **"Connect to Git"** → Skip this (we're using direct upload via GitHub Actions)
4. Or click **"Direct Upload"** and just note the project name

**Note**: The GitHub Actions workflow will automatically create a project named `openamundsen-da` if it doesn't exist yet.

## Step 5: Update the Project Name (Optional)

If you want a different project name, edit [.github/workflows/deploy-docs.yml](.github/workflows/deploy-docs.yml):

```yaml
command: pages deploy docs/_site --project-name=your-custom-name
```

## Step 6: Update Your Documentation URL

After the first deployment, Cloudflare will give you a URL like:
- `https://openamundsen-da.pages.dev`

If you want to use a custom domain:
1. Go to your Cloudflare Pages project
2. Click **Custom domains** → **Set up a custom domain**
3. Follow the instructions
4. Update `url` in [docs/_config.yml](docs/_config.yml) to your custom domain

## Step 7: Deploy!

Once you've added the secrets to GitHub:

1. Commit and push your changes:
   ```bash
   git add .
   git commit -m "Add Cloudflare Pages deployment workflow"
   git push origin main
   ```

2. Go to your repository → **Actions** tab
3. You should see the "Deploy Docs to Cloudflare Pages" workflow running
4. Once complete, your docs will be live at `https://openamundsen-da.pages.dev`

## Automatic Deployments

From now on, whenever you push changes to the `docs/` folder on the `main` branch, GitHub Actions will automatically:
1. Build your Jekyll site
2. Deploy it to Cloudflare Pages

You can also manually trigger a deployment:
1. Go to **Actions** → **Deploy Docs to Cloudflare Pages**
2. Click **"Run workflow"**

## Troubleshooting

### Workflow fails with "Project not found"
- Make sure you've created the Cloudflare Pages project first
- Or let the first workflow run create it automatically (it might fail the first time, then succeed on retry)

### Workflow fails with "Authentication error"
- Double-check your `CLOUDFLARE_API_TOKEN` and `CLOUDFLARE_ACCOUNT_ID` secrets
- Make sure the API token has the correct permissions (Cloudflare Pages → Edit)

### Site doesn't look right
- Check that the `url` and `baseurl` in [docs/_config.yml](docs/_config.yml) match your Cloudflare Pages URL
- The `baseurl` should be empty (`""`) for Cloudflare Pages

### Need to rebuild everything?
- Delete `docs/Gemfile.lock` locally
- Run `bundle install` in the `docs/` folder
- Commit and push the new `Gemfile.lock`

## Files Changed

This setup created/modified the following files:
- [.github/workflows/deploy-docs.yml](.github/workflows/deploy-docs.yml) - GitHub Actions workflow
- [docs/_config.yml](docs/_config.yml) - Updated URL for Cloudflare Pages
- [docs/Gemfile](docs/Gemfile) - Updated to remove GitHub Pages dependency
- This setup guide

## Adding Access Control (Optional - Protect Your Docs)

If you want to restrict who can view your documentation, you can use Cloudflare Access for free (up to 50 users):

### Step 1: Navigate to Zero Trust
1. Go to https://one.dash.cloudflare.com/
2. Click **"Access"** in the left sidebar
3. Click **"Applications"**

### Step 2: Create an Access Application
1. Click **"Add an application"**
2. Select **"Self-hosted"**

### Step 3: Configure the Application
Fill in these details:
- **Application name**: `openAMUNDSEN Docs`
- **Session duration**: `24h` (or your preference)
- **Application domain**: `openamundsen-da.pages.dev`
  - ⚠️ **Important**: Make sure it's exactly `openamundsen-da.pages.dev` (not duplicated like `openamundsen-da.openamundsen-da.pages.dev`)
- **Path**: Leave empty for entire site

### Step 4: Add an Access Policy
1. Click **"Add a policy"**
2. **Policy name**: `Allow specific users`
3. **Action**: `Allow`
4. **Configure rules** - Choose one:
   - **Emails**: Add specific email addresses (e.g., `you@example.com`)
   - **Email domains**: Allow entire domains (e.g., `uibk.ac.at`)
   - **Everyone**: For testing or public access

### Step 5: Save and Test
1. Click **"Add application"**
2. Wait 2-3 minutes for changes to propagate
3. Open your site in an **incognito/private window**
4. You should see a Cloudflare Access login screen!

### Troubleshooting Access
- **Site still accessible without login?**
  - Check the domain is exactly `openamundsen-da.pages.dev` (no duplicates)
  - Wait 5 minutes for global propagation
  - Try incognito mode
- **Can't access even with correct email?**
  - Verify your policy has **Action: Allow** (not Deny or Block)
  - Check your email is in the Include rules

## Next Steps

1. ✅ Add the GitHub secrets (Steps 2-3)
2. ✅ Create Cloudflare Pages project (Step 4)
3. ✅ Push your changes and watch them deploy
4. 🔒 (Optional) Set up Cloudflare Access for authentication
5. 🌐 (Optional) Set up a custom domain in Cloudflare
